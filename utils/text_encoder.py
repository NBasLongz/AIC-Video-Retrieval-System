import logging

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
        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=config.MODEL_TRUST_REMOTE_CODE,
        ).to(self.device)

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
        inputs = self.processor(
            text=[query],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                text_features = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                text_features = getattr(outputs, "text_embeds", None)
                if text_features is None:
                    text_features = getattr(outputs, "pooler_output", None)
                if text_features is None:
                    text_features = outputs.last_hidden_state[:, 0]

            if self.device == "cuda":
                text_features = text_features.cpu()
            return F.normalize(text_features, p=2, dim=-1).detach().numpy().astype(np.float32)
