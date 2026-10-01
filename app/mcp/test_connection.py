import asyncio
import traceback

from app.config import settings
from app.mcp.client import DataR2MCPClient


async def test_server(
    client: DataR2MCPClient,
    server_name: str,
):
    print()
    print("=" * 60)
    print(f"Testing: {server_name}")
    print("=" * 60)

    try:
        server = next(
            s
            for s in client.servers
            if s.name == server_name
        )

        tools = await client.list_tools(server)

        print("STATUS: CONNECTED")
        print(f"Tools found: {len(tools)}")
        print()

        for tool in tools:
            print(f"  - {tool['name']}")

            if tool.get("description"):
                print(f"    {tool['description']}")

        return True

    except Exception as exc:

        print("STATUS: FAILED")
        print(f"Exception type: {type(exc).__name__}")
        print(f"Error: {exc}")
        print()
        print("FULL TRACEBACK:")
        traceback.print_exc()

        # Python 3.11+ ExceptionGroup support
        if isinstance(exc, BaseExceptionGroup):

            print()
            print("=" * 60)
            print("SUB-EXCEPTIONS")
            print("=" * 60)

            for index, sub_exception in enumerate(
                exc.exceptions,
                start=1,
            ):
                print()
                print(
                    f"SUB-EXCEPTION {index}: "
                    f"{type(sub_exception).__name__}"
                )
                print(sub_exception)

                traceback.print_exception(
                    type(sub_exception),
                    sub_exception,
                    sub_exception.__traceback__,
                )

        return False


async def main():

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
        url="https://huggingface.co/mcp",
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
        url="https://www.kaggle.com/mcp",
        headers=kaggle_headers,
    )

    # ============================================================
    # Test
    # ============================================================

    await test_server(
        client,
        "huggingface",
    )

    await test_server(
        client,
        "kaggle",
    )


if __name__ == "__main__":
    asyncio.run(main())