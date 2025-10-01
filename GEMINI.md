# GEMINI.md

## Project Overview

This project is a Python-based MCP (Multi-Project Coordinator) server for providing comprehensive session management functionality for "Claude Code" sessions. It is designed to work across any project and provides features for session initialization, quality checkpoints, session cleanup, and real-time status monitoring.

The server is built on the `FastMCP` framework and includes a rich set of tools for various functionalities, including:

- **Session Management:** Tools for starting, checkpointing, and ending sessions, as well as for getting the current session status.
- **Crackerjack Integration:** Deep integration with the "Crackerjack" AI-driven Python development platform for tracking quality metrics, monitoring test results, and recognizing error patterns.
- **Memory and Reflection:** A sophisticated memory and reflection system that uses a local DuckDB database for conversation storage and ONNX for semantic search.
- **Natural Language Scheduling:** Tools for creating, listing, and canceling natural language reminders.
- **Interruption Management:** Tools for managing interruptions and preserving session context.
- **Multi-Project Coordination:** Support for coordinating across multiple projects, including creating project groups and searching for conversations across related projects.
- **Advanced Search:** Advanced search capabilities with faceted filtering and suggestions.
- **Git Worktree Management:** A comprehensive set of tools for managing git worktrees.

The project has a strong emphasis on code quality, with extensive configurations for various development tools like `ruff`, `pytest`, `coverage`, `pyright`, `bandit`, `vulture`, and `refurb`.

## Building and Running

### Installation

To install the project and its dependencies, use the following commands:

```bash
# Clone the repository
git clone https://github.com/lesleslie/session-mgmt-mcp.git
cd session-mgmt-mcp

# Install dependencies
uv sync --group dev
```

### Running the Server

The server can be run in either HTTP or STDIO mode.

**HTTP Mode:**

```bash
python -m session_mgmt_mcp.server --http
```

**STDIO Mode:**

```bash
python -m session_mgmt_mcp.server
```

The server can also be run using the `session-mgmt-mcp` script entry point defined in `pyproject.toml`:

```bash
session-mgmt-mcp --http
```

### Testing

The project uses `pytest` for testing. To run the tests, use the following command:

```bash
pytest
```

## Development Conventions

- **Code Style:** The project uses the "crackerjack" code style, which is enforced by `ruff`.
- **Typing:** The project uses type hints and is checked with `pyright`.
- **Testing:** The project has a comprehensive test suite that includes unit, functional, integration, performance, and security tests.
- **Modularity:** The project is organized into several modules, each responsible for a specific functionality. Tools are registered with the `FastMCP` server in a modular way.
- **Configuration:** The project uses a `pyproject.toml` file for configuration.
- **Database:** The project uses a local SQLite database for storing Crackerjack integration data and a DuckDB database for the memory and reflection system.
