from app.config import settings
from app.mcp.client import DataR2MCPClient


def create_mcp_client() -> DataR2MCPClient:
    """
    Create and configure the DataR² MCP client.

    The client connects to remote MCP servers.
    Server URLs are kept separate from the MCP client
    implementation so they can be changed without
    modifying client.py.
    """

    client = DataR2MCPClient()

    if settings.HF_MCP_URL:
        headers = {}

        if settings.HF_TOKEN:
            headers["Authorization"] = (
                f"Bearer {settings.HF_TOKEN}"
            )

        client.add_server(
            name="huggingface",
            url=settings.HF_MCP_URL,
            headers=headers,
        )

    if settings.KAGGLE_MCP_URL:
        headers = {}

        if settings.KAGGLE_TOKEN:
            headers["Authorization"] = (
                f"Bearer {settings.KAGGLE_TOKEN}"
            )

        client.add_server(
            name="kaggle",
            url=settings.KAGGLE_MCP_URL,
            headers=headers,
        )

    if settings.OPENML_MCP_URL:
        client.add_server(
            name="openml",
            url=settings.OPENML_MCP_URL,
        )

    if settings.UCI_MCP_URL:
        client.add_server(
            name="uci",
            url=settings.UCI_MCP_URL,
        )

    if settings.DATAGOV_MCP_URL:
        client.add_server(
            name="data.gov",
            url=settings.DATAGOV_MCP_URL,
        )

    return client