#!/usr/bin/env python3
"""Test the automatic working directory detection."""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from session_mgmt_mcp.tools.session_tools import _get_client_working_directory

def test_detection():
    """Test the working directory detection function."""
    print("🔍 Testing automatic working directory detection...")
    print("=" * 60)

    # Test each method individually
    print("🔍 Method 1: Environment variables...")
    import os
    for env_var in ["CLAUDE_WORKING_DIR", "CLIENT_PWD", "CLAUDE_PROJECT_DIR"]:
        value = os.environ.get(env_var)
        print(f"   • {env_var}: {value or 'Not set'}")

    print("\n🔍 Method 2: Temporary file...")
    temp_file = Path("/tmp/claude-git-working-dir")
    if temp_file.exists():
        content = temp_file.read_text().strip()
        print(f"   • File content: {content}")
        print(f"   • Ends with session-mgmt-mcp: {content.endswith('session-mgmt-mcp')}")
    else:
        print("   • File not found")

    print("\n🔍 Method 4: Recent git repositories...")
    for projects_dir in ["/Users/les/Projects"]:
        projects_path = Path(projects_dir)
        if projects_path.exists():
            print(f"   • Checking {projects_dir}")
            git_repos = []
            for repo_path in projects_path.iterdir():
                if repo_path.is_dir() and (repo_path / ".git").exists():
                    try:
                        mtime = repo_path.stat().st_mtime
                        git_repos.append((mtime, str(repo_path)))
                        print(f"     - Found git repo: {repo_path.name} (mtime: {mtime})")
                    except Exception as e:
                        print(f"     - Error with {repo_path}: {e}")

            if git_repos:
                git_repos.sort(reverse=True)
                print(f"   • Most recent: {git_repos[0][1]}")
                for i, (mtime, path) in enumerate(git_repos[:3]):
                    is_server = path.endswith("session-mgmt-mcp")
                    print(f"     {i+1}. {Path(path).name} ({'server' if is_server else 'candidate'})")

    detected_dir = _get_client_working_directory()

    if detected_dir:
        print(f"✅ Auto-detected working directory: {detected_dir}")

        # Validate it's a real directory
        if Path(detected_dir).exists():
            print(f"✅ Directory exists: {detected_dir}")

            # Check if it's a git repository
            if (Path(detected_dir) / ".git").exists():
                print(f"✅ Git repository detected: {detected_dir}")
            else:
                print(f"⚠️ Not a git repository: {detected_dir}")

            # Check if it's different from server directory
            server_dir = str(Path.cwd())
            if detected_dir != server_dir:
                print(f"✅ Different from server directory: {server_dir}")
            else:
                print(f"❌ Same as server directory: {server_dir}")

        else:
            print(f"❌ Directory does not exist: {detected_dir}")
    else:
        print("❌ No working directory detected")

    print("\n📊 Current environment:")
    print(f"   • Server directory: {Path.cwd()}")

    # Check the temporary file
    temp_file = Path("/tmp/claude-git-working-dir")
    if temp_file.exists():
        try:
            content = temp_file.read_text().strip()
            print(f"   • Temp file content: {content}")
        except Exception as e:
            print(f"   • Temp file error: {e}")
    else:
        print("   • Temp file: Not found")

    return detected_dir

if __name__ == "__main__":
    result = test_detection()
    if result:
        print(f"\n🎯 Final result: {result}")
    else:
        print("\n❌ No directory detected")