import asyncio
import json

from app.config import settings
from app.mcp.client import DataR2MCPClient


async def main():

    client = DataR2MCPClient()

    # ============================================================
    # Kaggle Authentication
    # ============================================================

    kaggle_headers = {}

    if settings.KAGGLE_TOKEN:
        kaggle_headers["Authorization"] = (
            f"Bearer {settings.KAGGLE_TOKEN}"
        )

    client.add_server(
        name="kaggle",
        url="https://www.kaggle.com/mcp",
        headers=kaggle_headers,
    )

    # ============================================================
    # Search Kaggle
    # ============================================================

    print("=" * 60)
    print("Searching Kaggle datasets...")
    print("=" * 60)

    try:

        result = await client.call_tool(
            server_name="kaggle",
            tool_name="search_datasets",
            arguments={
                "request": {
                    "search": "road traffic images",
                    "size": "All",
                    "fileType": "All",
                    "sortBy": "Relevance",
                    "pageSize": 10
                }
            },
        )

        print()
        print("SEARCH RESULT")
        print("=" * 60)

        print(
            json.dumps(
                result.model_dump(),
                indent=2,
                default=str,
            )
        )

    except Exception as exc:

        print()
        print("SEARCH FAILED")
        print("=" * 60)
        print(type(exc).__name__)
        print(str(exc))


if __name__ == "__main__":
    asyncio.run(main())