#!/usr/bin/env python3
"""Typer-based CLI for Session Management MCP Server.

Provides CLI interface matching crackerjack's server management pattern
with boolean options instead of subcommands.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404
import sys
import time
import warnings
from pathlib import Path
from typing import Any

# Suppress transformers warnings about PyTorch/TensorFlow
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*PyTorch.*TensorFlow.*Flax.*")

import psutil
import typer
from acb.console import console
from mcp_common.ui import ServerPanels

app = typer.Typer(
    name="session-buddy",
    help="Session Buddy MCP Server - CLI matching crackerjack pattern",
    no_args_is_help=True,
)

# Constants for typer.Option to fix FBT003 boolean positional values
DEFAULT_FALSE = False


def find_server_processes() -> list[psutil.Process]:
    """Find running session-buddy server processes."""
    processes = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            # Handle cmdline which can be None or list[str]
            cmdline_raw = proc.info["cmdline"]
            cmdline: list[str] = (
                cmdline_raw or [] if isinstance(cmdline_raw, list) else []
            )
            cmdline_str = " ".join(cmdline)

            # Look for various patterns that indicate a session-buddy server
            is_session_mcp = (
                "session_buddy" in cmdline_str or "session-buddy" in cmdline_str
            )

            # Check if it's running a server (various ways to start)
            is_server = (
                "server.py" in cmdline_str
                or "session_buddy.server" in cmdline_str
                or "--http" in cmdline_str
                or
                # Check if it's bound to our ports (for HTTP mode)
                any(
                    conn.laddr.port in {8678, 8677}
                    for conn in proc.net_connections()
                    if conn.laddr
                )
            )

            # Exclude CLI processes that are not servers
            is_cli_only = (
                "--start-mcp-server" in cmdline_str
                or "--stop-mcp-server" in cmdline_str
                or "--status" in cmdline_str
                or "--version" in cmdline_str
            )

            if is_session_mcp and is_server and not is_cli_only:
                processes.append(proc)

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def get_server_status() -> dict[str, Any]:
    """Get comprehensive server status information."""
    processes = find_server_processes()
    status: dict[str, Any] = {
        "running": len(processes) > 0,
        "process_count": len(processes),
        "processes": [],
        "http_port": 8678,
        "websocket_port": 8677,
    }

    for proc in processes:
        try:
            status["processes"].append(
                {
                    "pid": proc.pid,
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "cpu_percent": proc.cpu_percent(),
                    "create_time": proc.create_time(),
                },
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return status


def show_status() -> None:
    """Show comprehensive server status information."""
    server_status = get_server_status()

    # Create status table
    status_color = "green" if server_status["running"] else "red"
    status_text = "Running" if server_status["running"] else "Stopped"
    items = {
        "Status": f"[{status_color}]{status_text}[/{status_color}]",
        "Process Count": str(server_status["process_count"]),
        "HTTP Port": str(server_status["http_port"]),
        "WebSocket Port": str(server_status["websocket_port"]),
    }
    if server_status["running"]:
        items["HTTP Endpoint"] = f"http://127.0.0.1:{server_status['http_port']}/mcp"
        items["WebSocket Monitor"] = f"ws://127.0.0.1:{server_status['websocket_port']}"
    ServerPanels.config_table("Session Management MCP Server Status", items)

    # Show process details if running
    if server_status["processes"]:
        ServerPanels.process_list(server_status["processes"])


def start_mcp_server(
    port: int | None = None,
    websocket_port: int | None = None,
    verbose: bool = False,
) -> bool:
    """Start the session management MCP server."""
    server_status = get_server_status()

    if server_status["running"]:
        ServerPanels.status_panel(
            title="Server Status",
            status_text=f"[yellow]⚠️  Server already running with {server_status['process_count']} process(es)[/yellow]",
            severity="warning",
        )

        console.print("[yellow]⚠️  Stopping existing server first...[/yellow]")
        stop_mcp_server()
        time.sleep(2)

    console.print("[blue]🚀 Starting Session Management MCP Server...[/blue]")

    try:
        # Start server in HTTP mode so it can be properly detected by process monitoring
        # Use the configured ports or defaults
        http_port = port or 8678
        ws_port = websocket_port or 8677

        cmd = [
            sys.executable,
            "-c",
            f"from session_buddy.server import main; main(http_mode=True, http_port={http_port})",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None,
            start_new_session=True,
        )

        # Wait a moment for server to start and bind to ports
        console.print("[cyan]⏳ Waiting for server to start...[/cyan]")
        time.sleep(4)

        # Check if process is still running
        if process.poll() is not None:  # Process has exited
            console.print(
                f"[red]❌ Server process exited with code {process.returncode}[/red]",
            )
            if not verbose:
                console.print(
                    "[yellow]💡 Try --verbose flag to see startup errors[/yellow]",
                )
            raise typer.Exit(1)

        # Check if server is detected by our process monitoring
        final_status = get_server_status()
        if final_status["running"]:
            ServerPanels.status_panel(
                title="Session Management MCP Server",
                status_text="[green]✅ Server started successfully![/green]",
                items={
                    "HTTP Endpoint": f"http://127.0.0.1:{http_port}/mcp",
                    "WebSocket Monitor": f"ws://127.0.0.1:{ws_port}",
                    "Process ID": str(final_status["processes"][0]["pid"]),
                },
                severity="success",
            )
        else:
            ServerPanels.status_panel(
                title="Warning",
                status_text="[yellow]⚠️ Server process is running but not responding on expected ports[/yellow]",
                items={
                    "Process ID": str(process.pid),
                    "Mode": "STDIO?",
                },
                severity="warning",
            )

    except Exception as e:
        console.print(f"[red]❌ Failed to start server: {e}[/red]")
        if not verbose:
            console.print("[yellow]💡 Try --verbose flag for more details[/yellow]")
        raise typer.Exit(1)

    console.print("[green]✅ Server started successfully[/green]")
    return True


def stop_mcp_server() -> bool:
    """Stop all running session management MCP servers."""
    processes = find_server_processes()

    if not processes:
        ServerPanels.status_panel(
            title="Server Status",
            status_text="[yellow]⚠️ No running server processes found[/yellow]",
            description="Server may already be stopped or running in STDIO mode.",
            severity="warning",
        )
        return True  # Not an error if already stopped

    console.print(f"[blue]🛑 Stopping {len(processes)} server process(es)...[/blue]")

    stopped_count = 0
    for proc in processes:
        try:
            console.print(f"[cyan]Stopping process {proc.pid}...[/cyan]")

            # Try graceful termination first
            proc.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                proc.wait(timeout=5)
                console.print(
                    f"[green]✅ Process {proc.pid} terminated gracefully[/green]",
                )
                stopped_count += 1
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown failed
                console.print(
                    f"[yellow]⚡ Force killing process {proc.pid}...[/yellow]",
                )
                proc.kill()
                proc.wait()
                console.print(f"[green]✅ Process {proc.pid} force killed[/green]")
                stopped_count += 1

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            console.print(f"[red]❌ Failed to stop process {proc.pid}: {e}[/red]")

    if stopped_count > 0:
        ServerPanels.status_panel(
            title="Server Stopped",
            status_text=f"[green]✅ Successfully stopped {stopped_count} process(es)[/green]",
            severity="success",
        )
    else:
        console.print("[red]❌ Failed to stop any processes[/red]")
        return False

    return True


def restart_mcp_server(
    port: int | None = None,
    websocket_port: int | None = None,
    verbose: bool = False,
) -> bool:
    """Restart the session management MCP server (stop and start)."""
    ServerPanels.status_panel(
        title="Server Restart",
        status_text="[blue]🔄 Restarting Session Management MCP Server...[/blue]",
        severity="info",
    )

    # Stop existing servers
    console.print("[cyan]📴 Stopping existing servers...[/cyan]")
    stop_mcp_server()

    # Wait for cleanup
    console.print("[cyan]⏳ Waiting for cleanup...[/cyan]")
    time.sleep(2)

    # Start new server
    console.print("[cyan]🚀 Starting fresh server instance...[/cyan]")
    return start_mcp_server(port=port, websocket_port=websocket_port, verbose=verbose)


def show_version() -> None:
    """Show version information."""
    ServerPanels.status_panel(
        title="Version Information",
        status_text="[cyan]Session Management MCP Server[/cyan]",
        items={
            "Version": "2.0.0",
            "Description": "FastMCP-based server for Claude session management",
        },
        severity="info",
    )


def show_logs(lines: int = 50, follow: bool = False) -> None:
    """Show server logs."""
    log_dir = Path.home() / ".cache" / "claude" / "session_management"
    log_pattern = "session_management_*.log"

    if not log_dir.exists():
        console.print("[red]❌ Log directory not found[/red]")
        raise typer.Exit(1)

    log_files = list(log_dir.glob(log_pattern))
    if not log_files:
        console.print("[red]❌ No log files found[/red]")
        raise typer.Exit(1)

    # Get the most recent log file
    latest_log = max(log_files, key=lambda f: f.stat().st_mtime)
    console.print(f"[cyan]📄 Showing logs from: {latest_log}[/cyan]")

    try:
        if follow:
            subprocess.run(
                ["tail", "-f", "-n", str(lines), str(latest_log)],
                check=False,
            )
        else:
            subprocess.run(["tail", "-n", str(lines), str(latest_log)], check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]📝 Log following stopped[/yellow]")
    except FileNotFoundError:
        console.print("[red]❌ 'tail' command not found[/red]")


def show_config() -> None:
    """Show current server configuration."""
    items: dict[str, str] = {}

    # Read configuration from pyproject.toml
    try:
        import tomli

        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        if pyproject_path.exists():
            with pyproject_path.open("rb") as f:
                data = tomli.load(f)

            config = data.get("tool", {}).get("session-buddy", {})
            items["HTTP Port"] = str(config.get("mcp_http_port", 8678))
            items["HTTP Host"] = config.get("mcp_http_host", "127.0.0.1")
            items["WebSocket Port"] = str(config.get("websocket_monitor_port", 8677))
            items["HTTP Enabled"] = str(config.get("http_enabled", True))

        else:
            items["Status"] = "[red]pyproject.toml not found[/red]"

    except ImportError:
        items["Status"] = "[red]tomli not available[/red]"
    except Exception as e:
        items["Error"] = f"[red]{e}[/red]"

    ServerPanels.config_table("Server Configuration", items)


@app.command()
def main(
    start_mcp_server: bool = typer.Option(
        DEFAULT_FALSE,
        "--start-mcp-server",
        help="Start MCP server for session management",
    ),
    stop_mcp_server: bool = typer.Option(
        DEFAULT_FALSE,
        "--stop-mcp-server",
        help="Stop all running MCP servers",
    ),
    restart_mcp_server: bool = typer.Option(
        DEFAULT_FALSE,
        "--restart-mcp-server",
        help="Restart MCP server (stop and start again)",
    ),
    status: bool = typer.Option(
        DEFAULT_FALSE,
        "--status",
        help="Show comprehensive server status information",
    ),
    version: bool = typer.Option(
        DEFAULT_FALSE,
        "--version",
        help="Show version information",
    ),
    config: bool = typer.Option(
        DEFAULT_FALSE,
        "--config",
        help="Show current server configuration",
    ),
    logs: bool = typer.Option(
        DEFAULT_FALSE,
        "--logs",
        help="Show server logs",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="HTTP server port (for start/restart)",
    ),
    websocket_port: int | None = typer.Option(
        None,
        "--websocket-port",
        help="WebSocket monitor port (for start/restart)",
    ),
    verbose: bool = typer.Option(
        DEFAULT_FALSE,
        "--verbose",
        help="Enable verbose output",
    ),
) -> None:
    """Session Management MCP Server - CLI matching crackerjack pattern."""
    # Handle server management commands (like crackerjack)
    if start_mcp_server:
        start_mcp_server_func(port=port, websocket_port=websocket_port, verbose=verbose)
        return

    if stop_mcp_server:
        stop_mcp_server_func()
        return

    if restart_mcp_server:
        restart_mcp_server_func(
            port=port,
            websocket_port=websocket_port,
            verbose=verbose,
        )
        return

    # Handle status/info commands
    if status:
        show_status()
        return

    if version:
        show_version()
        return

    if config:
        show_config()
        return

    if logs:
        show_logs()
        return

    # If no options provided, show help
    console.print(
        "[yellow]No command specified. Use --help for available options.[/yellow]",
    )


# Rename functions to avoid conflicts with parameter names
start_mcp_server_func = start_mcp_server
stop_mcp_server_func = stop_mcp_server
restart_mcp_server_func = restart_mcp_server


if __name__ == "__main__":
    app()
