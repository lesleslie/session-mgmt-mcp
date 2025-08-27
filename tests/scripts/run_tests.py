#!/usr/bin/env python3
"""
Comprehensive test runner for session-mgmt-mcp.

This script provides a unified interface for running all test suites
with coverage reporting, quality metrics, and comprehensive output.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --quick            # Run quick smoke tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --performance      # Run only performance tests
    python run_tests.py --security         # Run only security tests
    python run_tests.py --coverage-only    # Generate coverage report only
    python run_tests.py --no-coverage      # Skip coverage reporting
    python run_tests.py --parallel         # Run tests in parallel
    python run_tests.py --verbose          # Verbose output
"""

import argparse
import sys
import subprocess
from pathlib import Path
import json
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.utils.test_runner import TestRunner
from tests.utils.test_data_manager import cleanup_test_data

def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for session-mgmt-mcp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests with coverage
  python run_tests.py --quick            # Quick smoke tests
  python run_tests.py --unit --verbose   # Verbose unit tests
  python run_tests.py --no-coverage      # All tests without coverage
  python run_tests.py --performance      # Performance tests only
        """
    )
    
    # Test suite selection
    suite_group = parser.add_mutually_exclusive_group()
    suite_group.add_argument(
        '--all', action='store_true', default=True,
        help='Run all test suites (default)'
    )
    suite_group.add_argument(
        '--quick', action='store_true',
        help='Run quick smoke tests only'
    )
    suite_group.add_argument(
        '--unit', action='store_true',
        help='Run unit tests only'
    )
    suite_group.add_argument(
        '--integration', action='store_true',
        help='Run integration tests only'
    )
    suite_group.add_argument(
        '--performance', action='store_true',
        help='Run performance tests only'
    )
    suite_group.add_argument(
        '--security', action='store_true',
        help='Run security tests only'
    )
    
    # Coverage options
    coverage_group = parser.add_mutually_exclusive_group()
    coverage_group.add_argument(
        '--coverage', action='store_true', default=True,
        help='Enable coverage reporting (default)'
    )
    coverage_group.add_argument(
        '--no-coverage', action='store_true',
        help='Disable coverage reporting'
    )
    coverage_group.add_argument(
        '--coverage-only', action='store_true',
        help='Generate coverage report only (no test execution)'
    )
    
    # Execution options
    parser.add_argument(
        '--parallel', action='store_true',
        help='Run tests in parallel'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Quiet output (minimal)'
    )
    parser.add_argument(
        '--timeout', type=int, default=600,
        help='Test timeout in seconds (default: 600)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir', type=Path, default=project_root / 'test_reports',
        help='Output directory for reports (default: test_reports/)'
    )
    parser.add_argument(
        '--json-output', type=Path,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--no-cleanup', action='store_true',
        help='Skip cleanup of test data (for debugging)'
    )
    
    # Quality thresholds
    parser.add_argument(
        '--min-coverage', type=float, default=85.0,
        help='Minimum coverage percentage required (default: 85.0)'
    )
    parser.add_argument(
        '--fail-on-coverage', action='store_true',
        help='Fail if coverage is below minimum'
    )
    
    return parser

def print_banner():
    """Print test runner banner"""
    print("=" * 70)
    print("🧪 Session Management MCP - Comprehensive Test Harness")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {project_root}")
    print()

def print_summary(results: dict, args):
    """Print test execution summary"""
    print("\n" + "=" * 70)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 70)
    
    summary = results.get('summary', {})
    
    # Overall status
    status = summary.get('overall_status', 'UNKNOWN')
    status_emoji = '✅' if status == 'PASSED' else '❌'
    print(f"Overall Status: {status_emoji} {status}")
    print(f"Execution Time: {results.get('execution_time', 0):.2f} seconds")
    print(f"Quality Score: {summary.get('quality_score', 0):.1f}/100.0")
    
    # Test suite results
    print(f"\n🧪 Test Suite Results:")
    test_suites = summary.get('test_suites', {})
    for suite_name, suite_info in test_suites.items():
        status_emoji = '✅' if suite_info['status'] == 'PASSED' else '❌'
        print(f"  {status_emoji} {suite_name.capitalize()}: "
              f"{suite_info['test_count']} tests, "
              f"{suite_info['failure_count']} failures, "
              f"{suite_info['execution_time']:.2f}s")
    
    # Coverage results
    if not args.no_coverage and not args.coverage_only:
        print(f"\n📈 Coverage Results:")
        coverage = summary.get('coverage_summary', {})
        line_cov = coverage.get('line_coverage', 0)
        branch_cov = coverage.get('branch_coverage', 0)
        target = coverage.get('coverage_target', args.min_coverage)
        meets_target = coverage.get('meets_target', False)
        
        target_emoji = '✅' if meets_target else '❌'
        print(f"  Line Coverage: {line_cov:.1f}% {target_emoji}")
        print(f"  Branch Coverage: {branch_cov:.1f}%")
        print(f"  Coverage Target: {target:.1f}%")
    
    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"  {i}. {rec}")
        if len(recommendations) > 5:
            print(f"  ... and {len(recommendations) - 5} more")
    
    # Output locations
    print(f"\n📂 Output Locations:")
    print(f"  Test Reports: {args.output_dir}")
    if not args.no_coverage and not args.coverage_only:
        print(f"  Coverage HTML: {project_root / 'htmlcov' / 'index.html'}")
        print(f"  Coverage XML: {project_root / 'coverage.xml'}")
    if args.json_output:
        print(f"  JSON Results: {args.json_output}")

def check_dependencies():
    """Check if required dependencies are available"""
    required_commands = ['pytest', 'coverage']
    missing = []
    
    for cmd in required_commands:
        try:
            result = subprocess.run(
                [cmd, '--version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                missing.append(cmd)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            missing.append(cmd)
    
    if missing:
        print(f"❌ Missing required dependencies: {', '.join(missing)}")
        print("Please install them with: pip install -r requirements-test.txt")
        return False
    
    return True

def main():
    """Main entry point"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.quiet:
        print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    
    # Initialize test runner
    runner = TestRunner(project_root)
    
    try:
        start_time = time.time()
        
        if args.coverage_only:
            # Generate coverage report only
            if not args.quiet:
                print("📈 Generating coverage report...")
            results = {'coverage': runner.generate_coverage_report()}
            
        elif args.quick:
            # Run quick tests
            if not args.quiet:
                print("⚡ Running quick smoke tests...")
            results = runner.run_quick_tests()
            
        elif args.unit:
            # Run unit tests only
            if not args.quiet:
                print("🔬 Running unit tests...")
            results = {
                'test_results': {
                    'unit': runner.run_test_suite(
                        'unit', 
                        coverage=not args.no_coverage,
                        parallel=args.parallel,
                        verbose=args.verbose
                    )
                }
            }
            
        elif args.integration:
            # Run integration tests only
            if not args.quiet:
                print("🔗 Running integration tests...")
            results = {
                'test_results': {
                    'integration': runner.run_test_suite(
                        'integration',
                        coverage=not args.no_coverage,
                        parallel=args.parallel,
                        verbose=args.verbose
                    )
                }
            }
            
        elif args.performance:
            # Run performance tests only
            if not args.quiet:
                print("🚀 Running performance tests...")
            results = {
                'test_results': {
                    'performance': runner.run_test_suite(
                        'performance',
                        coverage=not args.no_coverage,
                        parallel=False,  # Performance tests should not run in parallel
                        verbose=args.verbose
                    )
                }
            }
            
        elif args.security:
            # Run security tests only
            if not args.quiet:
                print("🔒 Running security tests...")
            results = {
                'test_results': {
                    'security': runner.run_test_suite(
                        'security',
                        coverage=not args.no_coverage,
                        parallel=args.parallel,
                        verbose=args.verbose
                    )
                }
            }
            
        else:
            # Run all tests (default)
            if not args.quiet:
                print("🧪 Running comprehensive test suite...")
            results = runner.run_all_tests(
                coverage=not args.no_coverage,
                parallel=args.parallel,
                verbose=args.verbose
            )
        
        # Calculate execution time
        results['execution_time'] = time.time() - start_time
        
        # Generate summary for single suite runs
        if 'summary' not in results and 'test_results' in results:
            results['summary'] = runner.generate_test_summary(results)
        
        # Save JSON output if requested
        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Print summary unless quiet
        if not args.quiet:
            print_summary(results, args)
        
        # Check for failures and coverage
        failed_tests = False
        coverage_failed = False
        
        # Check test results
        for suite_name, suite_results in results.get('test_results', {}).items():
            if not suite_results.get('success', False):
                failed_tests = True
                break
        
        # Check coverage if enabled and threshold set
        if args.fail_on_coverage and not args.no_coverage:
            coverage_percent = results.get('coverage', {}).get('summary', {}).get('coverage_percent', 0)
            if coverage_percent < args.min_coverage:
                coverage_failed = True
                if not args.quiet:
                    print(f"\n❌ Coverage below minimum: {coverage_percent:.1f}% < {args.min_coverage:.1f}%")
        
        # Exit with appropriate code
        if failed_tests or coverage_failed:
            if not args.quiet:
                print("\n❌ Tests failed or coverage insufficient")
            sys.exit(1)
        else:
            if not args.quiet:
                print("\n✅ All tests passed successfully!")
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup test data unless requested to keep it
        if not args.no_cleanup:
            try:
                cleanup_test_data()
            except Exception as e:
                if not args.quiet:
                    print(f"⚠️  Warning: Failed to cleanup test data: {e}")

if __name__ == "__main__":
    main()