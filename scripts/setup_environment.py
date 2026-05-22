"""
Script to check and setup database services for the video retrieval system.

Usage:
    python scripts/setup_environment.py --check
    python scripts/setup_environment.py --create-dirs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from backend import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def check_milvus():
    """Check if Milvus is accessible."""
    try:
        from pymilvus import connections, utility
        
        logger.info("Checking Milvus connection...")
        connections.connect(
            "default",
            host=config.MILVUS_HOST,
            port=config.MILVUS_PORT,
            timeout=5
        )
        
        # Try to list collections
        collections = utility.list_collections()
        logger.info(f"✓ Milvus is accessible at {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        logger.info(f"  Existing collections: {collections}")
        
        connections.disconnect("default")
        return True
    except Exception as e:
        logger.error(f"✗ Milvus connection failed: {e}")
        logger.error(f"  Make sure Milvus is running on {config.MILVUS_HOST}:{config.MILVUS_PORT}")
        logger.error(f"  Run: docker compose up -d")
        return False


def check_mongodb():
    """Check if MongoDB is accessible (optional - used by Milvus for metadata)."""
    # MongoDB is not directly configured in config.py
    # It's used internally by Milvus via Docker
    logger.info("Checking MongoDB...")
    logger.info("  ⓘ MongoDB runs via Docker (used by Milvus internally)")
    logger.info("  ⓘ No direct connection check needed")
    return True


def check_elasticsearch():
    """Check if Elasticsearch is accessible."""
    try:
        from elasticsearch import Elasticsearch
        
        logger.info("Checking Elasticsearch connection...")
        
        host = {
            "host": config.ELASTIC_HOST,
            "port": int(config.ELASTIC_PORT),
            "scheme": config.ELASTIC_SCHEME,
        }
        es = Elasticsearch(hosts=[host], request_timeout=5)
        
        # Check cluster health
        health = es.cluster.health()
        logger.info(f"✓ Elasticsearch is accessible at {config.ELASTIC_SCHEME}://{config.ELASTIC_HOST}:{config.ELASTIC_PORT}")
        logger.info(f"  Cluster status: {health.get('status', 'unknown')}")
        
        # List indices
        indices = es.indices.get_alias(index="*")
        logger.info(f"  Existing indices: {list(indices.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Elasticsearch connection failed: {e}")
        logger.error(f"  Make sure Elasticsearch is running")
        logger.error(f"  Run: docker compose up -d")
        return False


def create_directories():
    """Create necessary data directories if they don't exist."""
    logger.info("Creating necessary directories...")
    
    directories = [
        config.VIDEOS_DIR,
        config.KEYFRAMES_DIR,
        os.path.join(config.KEYFRAMES_DIR, "maps"),
        config.EMBEDDINGS_DIR,
        config.TRANSCRIPTS_DIR,
        config.OCR_RESULTS_DIR,
        config.CAPTIONS_DIR,
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ Created: {directory}")
        else:
            logger.info(f"  - Already exists: {directory}")
    
    logger.info("Directory setup complete!")


def check_python_packages():
    """Check if required Python packages are installed."""
    logger.info("Checking Python packages...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("open_clip", "OpenCLIP"),
        ("pymilvus", "Milvus client"),
        ("pymongo", "MongoDB client"),
        ("elasticsearch", "Elasticsearch client"),
        ("flask", "Flask web framework"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
    ]
    
    missing = []
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            logger.info(f"  ✓ {display_name}")
        except ImportError:
            logger.error(f"  ✗ {display_name} not installed")
            missing.append(package_name)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install them with: pip install -r requirements.txt")
        return False
    
    logger.info("All required packages are installed!")
    return True


def check_config():
    """Check configuration values."""
    logger.info("Checking configuration...")
    
    logger.info(f"  Milvus: {config.MILVUS_HOST}:{config.MILVUS_PORT}")
    logger.info(f"  Elasticsearch: {config.ELASTIC_SCHEME}://{config.ELASTIC_HOST}:{config.ELASTIC_PORT}")
    logger.info(f"  Visual Model: {config.VISUAL_MODEL_PROVIDER}:{config.VISUAL_MODEL_NAME} ({config.VISUAL_MODEL_PRETRAINED})")
    logger.info(f"  OCR Engine: {config.OCR_ENGINE}")
    logger.info(f"  Reranker: {config.RERANK_MODEL_PROVIDER}:{config.RERANK_MODEL_NAME}")
    logger.info(f"  Vector Dimension: {config.VECTOR_DIMENSION}")
    logger.info(f"  Videos Directory: {config.VIDEOS_DIR}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Setup and check environment for video retrieval system"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check database connections and configuration",
    )
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create necessary data directories",
    )
    parser.add_argument(
        "--check-packages",
        action="store_true",
        help="Check if required Python packages are installed",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks and setup",
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    success = True
    
    if args.all or args.check_packages:
        if not check_python_packages():
            success = False
        print()
    
    if args.all or args.create_dirs:
        create_directories()
        print()
    
    if args.all or args.check:
        check_config()
        print()
        
        logger.info("=" * 60)
        logger.info("Checking database services...")
        logger.info("=" * 60)
        
        milvus_ok = check_milvus()
        print()
        
        es_ok = check_elasticsearch()
        print()
        
        logger.info("=" * 60)
        if milvus_ok and es_ok:
            logger.info("✓ All services are ready!")
        else:
            logger.error("✗ Some services are not available")
            logger.error("  Please start the required services and try again")
            success = False
        logger.info("=" * 60)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
