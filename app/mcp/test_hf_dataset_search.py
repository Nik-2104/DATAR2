import asyncio
import json

from app.mcp.registry import create_mcp_client


async def main():

    client = create_mcp_client()

    print("=" * 60)
    print("SEARCHING HUGGING FACE DATASETS")
    print("=" * 60)

    try:

        result = await client.call_tool(
            server_name="huggingface",
            tool_name="hub_repo_search",
            arguments={
                "query": "road traffic images",
                "repo_types": ["dataset"],
                "sort": "downloads",
                "limit": 10,
            },
        )

        print()
        print("SEARCH RESULT")
        print("=" * 60)

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

    except Exception as exc:

        print()
        print("SEARCH FAILED")
        print("=" * 60)
        print(f"Exception: {type(exc).__name__}")
        print(f"Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())