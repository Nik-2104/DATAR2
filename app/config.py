from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ============================================================
    # Gemini
    # ============================================================

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ============================================================
    # MCP Servers
    # ============================================================

    # Hugging Face MCP
    HF_TOKEN: str | None = None

    # Kaggle MCP
    KAGGLE_TOKEN: str | None = None

    # ============================================================
    # MCP Server URLs
    # ============================================================

    HF_MCP_URL: str = "https://huggingface.co/mcp"
    KAGGLE_MCP_URL: str = "https://www.kaggle.com/mcp"

    OPENML_MCP_URL: str = ""
    UCI_MCP_URL: str = ""
    DATAGOV_MCP_URL: str = ""

    # ============================================================
    # Application
    # ============================================================

    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # ============================================================
    # Directories
    # ============================================================

    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )



settings = Settings()
