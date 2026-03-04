import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    firecrawl_api_key: str | None = None
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "gemma3:1b"
    
    # Model Configuration
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    vector_dim: int = 768  # specific for BAAI/bge-base-en-v1.5
    
    # Retrieval Configuration
    top_k: int = 3
    batch_size: int = 512
    rerank_top_k: int = 3
    
    # Database Configuration
    milvus_db_path: str = "./data/milvus_binary.db"
    collection_name: str = "paralegal_agent"
    
    # Data Configuration
    docs_path: str = "./data/raft.pdf"

    # Cache Configuration
    hf_cache_dir: str = "./cache/hf_cache"
    
    # LLM settings
    temperature: float = 0.6
    max_tokens: int = 1000
    
    model_config: SettingsConfigDict = SettingsConfigDict(
    env_file=".env",
    case_sensitive=False,
    )

def model_post_init(self, __context) -> None:
    # Create necessary directories
    Path(self.milvus_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(self.hf_cache_dir).mkdir(parents=True, exist_ok=True)

# Global settings instance
settings = Settings()