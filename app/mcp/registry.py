from app.config import settings
from app.mcp.client import DataR2MCPClient


def create_mcp_client() -> DataR2MCPClient:
    """
    Create the central MCP client and register
    all currently configured MCP servers.
    """

    client = DataR2MCPClient()

    # ============================================================
    # Hugging Face
    # ============================================================

    hf_headers = {}

    if settings.HF_TOKEN:
        hf_headers["Authorization"] = (
            f"Bearer {settings.HF_TOKEN}"
        )

    client.add_server(
        name="huggingface",
        url=settings.HF_MCP_URL,
        headers=hf_headers,
    )

    # ============================================================
    # Kaggle
    # ============================================================

    kaggle_headers = {}

    if settings.KAGGLE_TOKEN:
        kaggle_headers["Authorization"] = (
            f"Bearer {settings.KAGGLE_TOKEN}"
        )

    client.add_server(
        name="kaggle",
        url=settings.KAGGLE_MCP_URL,
        headers=kaggle_headers,
    )

    return client