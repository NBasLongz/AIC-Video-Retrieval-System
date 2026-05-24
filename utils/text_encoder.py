import logging
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backend import config

logger = logging.getLogger(__name__)


class TextEncoder:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.provider = config.VISUAL_MODEL_PROVIDER
        self.model_name = config.VISUAL_MODEL_NAME
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._precomputed_tokens = None
        logger.info(
            "TextEncoder created (provider=%s, model=%s, lazy loading enabled)",
            self.provider,
            self.model_name,
        )

    def _ensure_loaded(self):
        """Lazy load the model only when needed."""
        if self._model is not None:
            return

        if self.provider == "openclip":
            self._load_openclip()
        elif self.provider in {"jina_clip", "jina"}:
            self._load_jina_clip()
        else:
            self._load_transformers()

        self._model.eval()

        # Precompute common query tokens for performance
        self._precomputed_tokens = {}
        logger.info("Text encoder loaded successfully.")

    def _load_openclip(self):
        import open_clip

        logger.info(
            "Loading OpenCLIP text model '%s' (%s) to device '%s'...",
            config.CLIP_MODEL_NAME,
            config.CLIP_PRETRAINED,
            self.device,
        )
        self._model, _, _ = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME,
            pretrained=config.CLIP_PRETRAINED,
        )

        # Remove visual encoder to save memory (we only need text encoder online).
        if hasattr(self._model, "visual"):
            del self._model.visual

        self._model = self._model.to(self.device)
        self._tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)

    def _load_transformers(self):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "VISUAL_MODEL_PROVIDER requires transformers. Install optional "
                "dependencies or set VISUAL_MODEL_PROVIDER=openclip."
            ) from exc

        logger.info(
            "Loading transformers text model '%s' to device '%s'...",
            self.model_name,
            self.device,
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=config.MODEL_TRUST_REMOTE_CODE,
        )
        if self.provider == "siglip2" and config.VISUAL_TEXT_ONLY_MODEL and config.VISUAL_STREAM_SAFE_LOAD:
            self._load_siglip2_text_streaming()
            return

        model_cls = AutoModel
        if self.provider == "siglip2" and config.VISUAL_TEXT_ONLY_MODEL:
            try:
                from transformers import Siglip2TextModel

                model_cls = Siglip2TextModel
                logger.info("Using Siglip2TextModel text-only loader to avoid loading the vision branch.")
            except ImportError:
                logger.warning("Siglip2TextModel is unavailable; falling back to AutoModel.")

        self._model = model_cls.from_pretrained(
            self.model_name,
            trust_remote_code=config.MODEL_TRUST_REMOTE_CODE,
            **self._from_pretrained_kwargs(),
        ).to(self.device)

    def _load_siglip2_text_streaming(self):
        try:
            from huggingface_hub import snapshot_download
            from transformers import Siglip2Config, Siglip2TextModel
        except ImportError as exc:
            raise ImportError("Streaming SigLIP2 text load requires huggingface_hub and transformers.") from exc

        snapshot_dir = Path(snapshot_download(self.model_name, local_files_only=True))
        checkpoint_path = snapshot_dir / "model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SigLIP2 checkpoint not found in local cache: {checkpoint_path}")

        logger.info(
            "Streaming SigLIP2 text weights from %s without memory-mapping the full checkpoint...",
            checkpoint_path,
        )
        full_config = Siglip2Config.from_pretrained(
            self.model_name,
            trust_remote_code=config.MODEL_TRUST_REMOTE_CODE,
            local_files_only=True,
        )
        dtype = self._torch_dtype()
        original_default_dtype = torch.get_default_dtype()
        try:
            if dtype is not None and dtype != "auto":
                torch.set_default_dtype(dtype)
            self._model = Siglip2TextModel(full_config.text_config)
        finally:
            torch.set_default_dtype(original_default_dtype)
        self._model = self._model.to(self.device)

        self._stream_safetensors_into_model(checkpoint_path, self._model)

    def _stream_safetensors_into_model(self, checkpoint_path: Path, model: torch.nn.Module):
        dtype_map = {
            "F64": np.float64,
            "F32": np.float32,
            "F16": np.float16,
            "BF16": np.uint16,
            "I64": np.int64,
            "I32": np.int32,
            "I16": np.int16,
            "I8": np.int8,
            "U8": np.uint8,
            "BOOL": np.bool_,
        }
        itemsize_map = {name: np.dtype(dtype).itemsize for name, dtype in dtype_map.items()}
        tensors = dict(model.named_parameters())
        tensors.update(dict(model.named_buffers()))
        loaded = 0
        missing = []

        with checkpoint_path.open("rb") as handle, torch.no_grad():
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len).decode("utf-8"))
            data_start = 8 + header_len

            for name, target in tensors.items():
                meta = header.get(name)
                if meta is None:
                    missing.append(name)
                    continue
                dtype_name = meta["dtype"]
                if dtype_name not in dtype_map:
                    raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for {name}")
                shape = tuple(int(value) for value in meta["shape"])
                if tuple(target.shape) != shape:
                    raise ValueError(f"Shape mismatch for {name}: checkpoint={shape} model={tuple(target.shape)}")
                start, end = (int(value) for value in meta["data_offsets"])
                expected_bytes = end - start
                itemsize = itemsize_map[dtype_name]
                total_elements = expected_bytes // itemsize
                source_dtype = dtype_map[dtype_name]

                flat_target = target.data.view(-1)
                max_bytes = 64 * 1024 * 1024
                max_elements = max(1, max_bytes // itemsize)
                for offset in range(0, total_elements, max_elements):
                    count = min(max_elements, total_elements - offset)
                    handle.seek(data_start + start + offset * itemsize)
                    raw = handle.read(count * itemsize)
                    array = np.frombuffer(raw, dtype=source_dtype, count=count)
                    if dtype_name == "BF16":
                        tensor = torch.from_numpy(array.astype(np.uint16)).view(torch.bfloat16)
                    else:
                        tensor = torch.from_numpy(array)
                    tensor = tensor.to(device=flat_target.device, dtype=flat_target.dtype)
                    flat_target[offset:offset + count].copy_(tensor)
                loaded += 1

        if missing:
            logger.warning("SigLIP2 streaming loader left %s model tensors at init values.", len(missing))
        logger.info("Streamed %s SigLIP2 text tensors into memory.", loaded)

    def _from_pretrained_kwargs(self):
        kwargs = {}
        dtype = self._torch_dtype()
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if config.VISUAL_LOW_CPU_MEM_USAGE:
            kwargs["low_cpu_mem_usage"] = True
        return kwargs

    def _torch_dtype(self):
        dtype = (config.VISUAL_MODEL_DTYPE or "").strip().lower()
        if dtype in {"", "none", "default", "float32", "fp32"}:
            return None
        if dtype in {"auto"}:
            return "auto"
        if dtype in {"float16", "fp16", "half"}:
            return torch.float16
        if dtype in {"bfloat16", "bf16"}:
            return torch.bfloat16
        logger.warning("Unsupported VISUAL_MODEL_DTYPE=%r; using model default dtype.", config.VISUAL_MODEL_DTYPE)
        return None

    def _feature_tensor(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs

        for attr_name in ("text_embeds", "image_embeds", "pooler_output"):
            value = getattr(outputs, attr_name, None)
            if value is not None:
                return value

        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is not None:
            return last_hidden[:, 0]

        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    return value

        raise TypeError(f"Cannot extract feature tensor from output type {type(outputs)!r}")

    def _load_jina_clip(self):
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError("jina-clip-v2 requires transformers. Install requirements.txt first.") from exc

        logger.info(
            "Loading Jina CLIP text model '%s' to device '%s'...",
            self.model_name,
            self.device,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            dtype="auto",
        ).to(self.device)
    
    @property
    def model(self):
        self._ensure_loaded()
        return self._model
    
    @property
    def tokenizer(self):
        self._ensure_loaded()
        return self._tokenizer

    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor
    
    @property
    def precomputed_tokens(self):
        self._ensure_loaded()
        return self._precomputed_tokens

    def encode(self, query: str):
        self._ensure_loaded()
        if self.provider == "openclip":
            return self._encode_openclip(query)
        if self.provider in {"jina_clip", "jina"}:
            return self._encode_jina_clip(query)
        return self._encode_transformers(query)

    def _encode_openclip(self, query: str):
        text_inputs = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            if self.device == "cuda":
                text_features = text_features.cpu()
            return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)

    def _encode_jina_clip(self, query: str):
        with torch.no_grad():
            text_features = self.model.encode_text(
                query,
                task="retrieval.query",
                truncate_dim=config.VISUAL_TRUNCATE_DIM,
            )
            if isinstance(text_features, torch.Tensor):
                if text_features.device.type != "cpu":
                    text_features = text_features.cpu()
                text_features = text_features.detach().numpy()
            text_features = torch.from_numpy(np.asarray(text_features, dtype=np.float32)).reshape(1, -1)
            return F.normalize(text_features, p=2, dim=-1).numpy().astype(np.float32)

    def _encode_transformers(self, query: str):
        text = query.lower() if self.provider == "siglip2" else query
        processor_kwargs = {
            "text": [text],
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
        }
        if self.provider == "siglip2":
            processor_kwargs["padding"] = "max_length"
            processor_kwargs["max_length"] = 64

        inputs = self.processor(
            **processor_kwargs,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                text_features = self._feature_tensor(self.model.get_text_features(**inputs))
            else:
                outputs = self.model(**inputs)
                text_features = self._feature_tensor(outputs)

            if self.device == "cuda":
                text_features = text_features.cpu()
            return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)
