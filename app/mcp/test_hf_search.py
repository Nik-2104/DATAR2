import asyncio
import json

from app.config import settings
from app.mcp.registry import create_mcp_client


async def main():

    client = create_mcp_client()

    server = next(
        (
            server
            for server in client.servers
            if server.name == "huggingface"
        ),
        None,
    )

    if server is None:
        print("Hugging Face server is not registered.")
        return

    print("=" * 60)
    print("HUGGING FACE MCP TOOLS")
    print("=" * 60)

    try:

        tools = await client.list_tools(server)

        for tool in tools:

            print()
            print("-" * 60)
            print(f"NAME: {tool['name']}")
            print("-" * 60)

            print(
                json.dumps(
                    tool,
                    indent=2,
                    default=str,
                )
            )

    except Exception as exc:

        print()
        print("FAILED")
        print("=" * 60)
        print(type(exc).__name__)
        print(str(exc))


if __name__ == "__main__":
    asyncio.run(main())
    