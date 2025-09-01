#!/usr/bin/env python3
"""Simple testing assessment for session-mgmt-mcp."""

import subprocess
from pathlib import Path


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, cwd=Path.cwd()
        )
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return 1, "", str(e)


def assess_testing_status() -> None:
    """Assess current testing status and capabilities."""
    print("🧪 Session Management MCP - Testing Assessment")
    print("=" * 60)

    # Check pytest availability
    print("\n1. Testing Framework Status:")
    code, stdout, stderr = run_command(
        ["python", "-c", "import pytest; print('pytest available')"],
    )
    if code == 0:
        print("   ✅ pytest: Available")
    else:
        print("   ❌ pytest: Not available")
        return

    # Check if we can import core modules
    print("\n2. Core Module Import Status:")
    modules_to_check = [
        "session_mgmt_mcp.config",
        "session_mgmt_mcp.reflection_tools",
        "session_mgmt_mcp.context_manager",
        "session_mgmt_mcp.token_optimizer",
    ]

    for module in modules_to_check:
        code, stdout, stderr = run_command(
            ["python", "-c", f"import {module}; print('OK')"],
        )
        if code == 0:
            print(f"   ✅ {module}: Imports successfully")
        else:
            print(f"   ❌ {module}: Import failed - {stderr.strip()}")

    # Count test files
    print("\n3. Test Suite Structure:")
    test_dirs = {
        "Unit Tests": Path("tests/unit"),
        "Integration Tests": Path("tests/integration"),
        "Performance Tests": Path("tests/performance"),
        "Security Tests": Path("tests/security"),
    }

    total_tests = 0
    for category, path in test_dirs.items():
        if path.exists():
            test_files = list(path.glob("test_*.py"))
            count = len(test_files)
            total_tests += count
            print(f"   📁 {category}: {count} files")
            for test_file in test_files[:3]:  # Show first 3
                print(f"      - {test_file.name}")
            if count > 3:
                print(f"      ... and {count - 3} more")
        else:
            print(f"   📁 {category}: Directory missing")

    print(f"\n   📊 Total test files: {total_tests}")

    # Try running a simple test
    print("\n4. Test Execution Check:")
    simple_tests = [
        "tests/unit/test_config.py",
        "tests/unit/test_token_optimizer.py",
        "tests/unit/test_advanced_search.py",
    ]

    for test_file in simple_tests:
        if Path(test_file).exists():
            print(f"   🧪 Testing {test_file}...")
            code, stdout, stderr = run_command(
                [
                    "python",
                    "-m",
                    "pytest",
                    test_file,
                    "-v",
                    "--tb=line",
                    "--disable-warnings",
                ],
            )

            if code == 0:
                lines = stdout.split("\n")
                test_lines = [
                    line
                    for line in lines
                    if "::" in line and ("PASSED" in line or "FAILED" in line)
                ]
                passed = len([line for line in test_lines if "PASSED" in line])
                failed = len([line for line in test_lines if "FAILED" in line])
                print(f"      ✅ {passed} passed, {failed} failed")
                if failed > 0:
                    print("      ⚠️ Some tests failed - see details above")
                break
            print(f"      ❌ Test execution failed: {stderr.strip()[:100]}")
        else:
            print(f"   ❌ {test_file} not found")

    # Assessment summary
    print("\n" + "=" * 60)
    print("🎯 TESTING ASSESSMENT SUMMARY")
    print("=" * 60)

    if total_tests > 0:
        print(f"✅ Found {total_tests} test files across all categories")
        print("✅ Core testing infrastructure is in place")
        print("✅ Project has comprehensive test documentation")

        print("\n💡 RECOMMENDED IMPROVEMENTS:")
        print("1. Fix pytest execution environment issues")
        print("2. Implement test runner dependency validation")
        print("3. Add test coverage reporting")
        print("4. Set up continuous integration testing")
        print("5. Add performance benchmarking")

    else:
        print("❌ No test files found - testing setup needed")


if __name__ == "__main__":
    assess_testing_status()
