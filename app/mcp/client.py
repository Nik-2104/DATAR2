from typing import Any

import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPServerConnection:
    """
    Represents one remote MCP server.
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        self.name = name
        self.url = url
        self.headers = headers or {}


class DataR2MCPClient:
    """
    Central MCP client for DataR².

    Uses the MCP Streamable HTTP transport.
    """

    def __init__(
        self,
        servers: list[MCPServerConnection] | None = None,
    ):
        self.servers = servers or []

    # ============================================================
    # Server Registration
    # ============================================================

    def add_server(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
    ):
        self.servers.append(
            MCPServerConnection(
                name=name,
                url=url,
                headers=headers,
            )
        )

    # ============================================================
    # List Tools
    # ============================================================

    async def list_tools(
        self,
        server: MCPServerConnection,
    ) -> list[dict[str, Any]]:

        async with httpx.AsyncClient(
            headers=server.headers,
            timeout=60.0,
        ) as http_client:

            async with streamable_http_client(
                url=server.url,
                http_client=http_client,
            ) as streams:

                read_stream, write_stream = streams

                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:

                    await session.initialize()

                    result = await session.list_tools()

                    tools = []

                    for tool in result.tools:
                        tools.append(
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "input_schema": tool.input_schema,
                            }
                        )

                    return tools

    # ============================================================
    # Discover Tools From All Servers
    # ============================================================

    async def discover_all_tools(self):

        results = {}

        for server in self.servers:

            try:

                tools = await self.list_tools(server)

                results[server.name] = tools

            except Exception as exc:

                results[server.name] = {
                    "error": str(exc)
                }

        return results

    # ============================================================
    # Call MCP Tool
    # ============================================================

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ):

        server = next(
            (
                server
                for server in self.servers
                if server.name == server_name
            ),
            None,
        )

        if server is None:
            raise ValueError(
                f"MCP server '{server_name}' "
                f"is not registered."
            )

        async with httpx.AsyncClient(
            headers=server.headers,
            timeout=120.0,
        ) as http_client:

            async with streamable_http_client(
                url=server.url,
                http_client=http_client,
            ) as streams:

                read_stream, write_stream = streams

                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:

                    await session.initialize()

                    result = await session.call_tool(
                        tool_name,
                        arguments or {},
                    )

                    return result

    # ============================================================
    # Search All Servers
    # ============================================================

    async def search_all(
        self,
        tool_arguments: dict[str, Any],
        tool_name: str = "search_datasets",
    ):

        results = []

        for server in self.servers:

            try:

                result = await self.call_tool(
                    server_name=server.name,
                    tool_name=tool_name,
                    arguments=tool_arguments,
                )

                results.append(
                    {
                        "server": server.name,
                        "result": result,
                    }
                )

            except Exception as exc:

                results.append(
                    {
                        "server": server.name,
                        "error": str(exc),
                    }
                )

        return results