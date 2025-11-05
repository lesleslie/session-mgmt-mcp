# Gemini Code Assistant Context

This document provides a comprehensive overview of the `session-mgmt-mcp` project, its structure, and development conventions to assist the Gemini code assistant in providing accurate and context-aware responses.

## Project Overview

The `session-mgmt-mcp` project is a Python-based MCP (Multi-Component Protocol) server that provides comprehensive session management functionality for "Claude Code" sessions. It is designed to work across any project, offering features like session initialization, quality checkpoints, session cleanup, and real-time status monitoring.

The server integrates with "Crackerjack" for code quality enforcement, uses a local DuckDB database with vector support for conversation memory and knowledge graph features, and leverages ONNX for local AI embeddings. The server is built using the `FastMCP` framework.

### Key Technologies

- **Programming Language:** Python 3.13+
- **Server Framework:** `fastmcp`
- **Database:** DuckDB (for conversation memory and knowledge graph)
- **Code Quality:** Crackerjack, Ruff (linting), Pytest (testing), Pyright (type checking)
- **Dependencies:** `pydantic`, `typer`, `onnxruntime`, `transformers`, and more (see `pyproject.toml`).
- **Building:** `hatchling`

## Building and Running

### Installation

To install the project and its dependencies, use `uv`:

```bash
# Install with all dependencies (development + testing)
uv sync --group dev

# Or install minimal production dependencies only
uv sync
```

### Running the Server

The server can be run in two modes:

1. **STDIO Mode (default):**

   ```bash
   python -m session_mgmt_mcp.server
   ```

   or using the script entry point:

   ```bash
   session-mgmt-mcp
   ```

1. **HTTP Mode:**

   ```bash
   python -m session_mgmt_mcp.server --http
   ```

   You can also specify a port:

   ```bash
   python -m session_mgmt_mcp.server --http --http-port 8000
   ```

### Running Tests

Tests are written using `pytest`. To run the test suite:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=session_mgmt_mcp
```

## Development Conventions

- **Coding Style:** The project follows the "Crackerjack" code style.
- **Linting:** `ruff` is used for linting. The configuration can be found in `pyproject.toml`.
- **Type Checking:** `pyright` is used for static type checking.
- **Testing:** Tests are located in the `tests/` directory and are written using `pytest`. The project has unit, functional, integration, and other types of tests.
- **Commits:** (No explicit convention found, but assume standard practices like conventional commits would be welcome).

## Key Files

- `README.md`: Provides a detailed overview of the project, its features, and how to use it.
- `pyproject.toml`: Defines the project's metadata, dependencies, and tool configurations (for `ruff`, `pytest`, `pyright`, etc.).
- `session_mgmt_mcp/server.py`: The main entry point for the MCP server. It initializes the `FastMCP` server and registers all the available tools.
- `session_mgmt_mcp/cli.py`: Defines a Typer-based command-line interface for managing the server (starting, stopping, checking status, etc.).
- `session_mgmt_mcp/tools/`: This directory likely contains the implementation of the various tool categories mentioned in the `README.md`.
- `tests/`: Contains all the tests for the project.

## Usage

The server can be managed via the command-line interface defined in `session_mgmt_mcp/cli.py`.

### CLI Commands

- `session-mgmt-mcp --start-mcp-server`: Starts the MCP server.
- `session-mgmt-mcp --stop-mcp-server`: Stops the MCP server.
- `session-mgmt-mcp --restart-mcp-server`: Restarts the MCP server.
- `session-mgmt-mcp --status`: Shows the server's status.
- `session-mgmt-mcp --version`: Shows the version of the server.
- `session-mgmt-mcp --config`: Shows the server's configuration.
- `session-mgmt-mcp --logs`: Shows the server's logs.

### Slash Commands (in Claude Code)

Once the server is running, it provides a set of slash commands in "Claude Code" for session management, memory search, and more. The primary commands are:

- `/session-mgmt:start`
- `/session-mgmt:checkpoint`
- `/session-mgmt:end`
- `/session-mgmt:status`

The server also creates convenient shortcuts for these commands (`/start`, `/checkpoint`, `/end`).
