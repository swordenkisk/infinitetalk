"""FastAPI Server"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch

from ..pipelines.inference_pipeline import InfiniteTalkPipeline
from ..core.sparse_frame_generator import SparseFrameGenerator, SparseAnchorConfig

# Global pipeline instance
pipeline = None

def load_pipeline(config_path: str = None):
    """Load the inference pipeline"""
    global pipeline
    # Implementation depends on model weights
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager"""
    # Startup
    load_pipeline()
    yield
    # Shutdown
    if pipeline:
        del pipeline
        torch.cuda.empty_cache()

def create_app() -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="InfiniteTalk API",
        description="Sparse-Frame Video Dubbing API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "pipeline_loaded": pipeline is not None}
        
    @app.get("/")
    async def root():
        return {
            "name": "InfiniteTalk API",
            "version": "1.0.0",
            "endpoints": ["/v1/video/generations", "/health"]
        }
        
    return app
