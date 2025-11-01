#!/usr/bin/env python3
"""
Automated test coverage reporting script for RTradez.

This script runs tests with coverage analysis and generates comprehensive reports.
"""

import subprocess
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import datetime


def run_command(cmd: List[str], description: str) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    if result.returncode != 0:
        print(f"❌ {description} failed!")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return result
    else:
        print(f"✅ {description} completed successfully")
        return result


def run_tests_with_coverage(test_args: List[str] = None) -> Dict[str, Any]:
    """Run tests with coverage analysis."""
    cmd = ["python", "-m", "pytest"]
    
    if test_args:
        cmd.extend(test_args)
    
    result = run_command(cmd, "Running tests with coverage")
    
    # Parse coverage output
    coverage_info = {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }
    
    # Extract coverage percentage from output
    for line in result.stdout.split('\n'):
        if 'TOTAL' in line and '%' in line:
            parts = line.split()
            for part in parts:
                if '%' in part and part.replace('%', '').replace('.', '').isdigit():
                    coverage_info["total_coverage"] = float(part.replace('%', ''))
                    break
    
    return coverage_info


def generate_coverage_badge(coverage_percentage: float) -> str:
    """Generate coverage badge color based on percentage."""
    if coverage_percentage >= 90:
        return "brightgreen"
    elif coverage_percentage >= 80:
        return "green" 
    elif coverage_percentage >= 70:
        return "yellow"
    elif coverage_percentage >= 60:
        return "orange"
    else:
        return "red"


def generate_coverage_summary() -> Dict[str, Any]:
    """Generate coverage summary from XML report."""
    coverage_file = Path("coverage.xml")
    
    if not coverage_file.exists():
        print("⚠️  Coverage XML file not found")
        return {}
    
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "packages": {},
            "overall": {
                "line_rate": float(root.attrib.get("line-rate", 0)) * 100,
                "branch_rate": float(root.attrib.get("branch-rate", 0)) * 100,
                "lines_covered": int(root.attrib.get("lines-covered", 0)),
                "lines_valid": int(root.attrib.get("lines-valid", 0)),
                "branches_covered": int(root.attrib.get("branches-covered", 0)),
                "branches_valid": int(root.attrib.get("branches-valid", 0))
            }
        }
        
        # Parse package details
        for package in root.findall(".//package"):
            package_name = package.attrib.get("name", "unknown")
            summary["packages"][package_name] = {
                "line_rate": float(package.attrib.get("line-rate", 0)) * 100,
                "branch_rate": float(package.attrib.get("branch-rate", 0)) * 100,
                "complexity": float(package.attrib.get("complexity", 0))
            }
        
        return summary
        
    except Exception as e:
        print(f"⚠️  Error parsing coverage XML: {e}")
        return {}


def save_coverage_report(summary: Dict[str, Any], output_file: str = "coverage_report.json"):
    """Save coverage report to JSON file."""
    output_path = Path(output_file)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Coverage report saved to {output_path}")


def print_coverage_summary(summary: Dict[str, Any]):
    """Print formatted coverage summary."""
    if not summary:
        print("❌ No coverage summary available")
        return
    
    overall = summary.get("overall", {})
    line_rate = overall.get("line_rate", 0)
    branch_rate = overall.get("branch_rate", 0)
    
    print("\n" + "="*60)
    print("📊 COVERAGE SUMMARY")
    print("="*60)
    print(f"Overall Line Coverage:   {line_rate:.2f}%")
    print(f"Overall Branch Coverage: {branch_rate:.2f}%")
    print(f"Lines Covered:           {overall.get('lines_covered', 0)}/{overall.get('lines_valid', 0)}")
    print(f"Branches Covered:        {overall.get('branches_covered', 0)}/{overall.get('branches_valid', 0)}")
    print(f"Badge Color:             {generate_coverage_badge(line_rate)}")
    
    # Package breakdown
    packages = summary.get("packages", {})
    if packages:
        print("\n📦 PACKAGE BREAKDOWN:")
        print("-" * 40)
        for package, stats in packages.items():
            print(f"{package:30} {stats['line_rate']:6.2f}%")
    
    print("="*60)


def check_coverage_threshold(coverage_percentage: float, threshold: float = 80.0) -> bool:
    """Check if coverage meets threshold."""
    if coverage_percentage >= threshold:
        print(f"✅ Coverage {coverage_percentage:.2f}% meets threshold of {threshold}%")
        return True
    else:
        print(f"❌ Coverage {coverage_percentage:.2f}% below threshold of {threshold}%")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run test coverage analysis")
    parser.add_argument("--threshold", type=float, default=80.0, 
                       help="Coverage threshold percentage (default: 80)")
    parser.add_argument("--output", default="coverage_report.json",
                       help="Output file for coverage report")
    parser.add_argument("--fail-under", action="store_true",
                       help="Fail if coverage is below threshold")
    parser.add_argument("--unit-only", action="store_true",
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true", 
                       help="Run only integration tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Prepare test arguments
    test_args = []
    if args.unit_only:
        test_args.extend(["-m", "unit"])
    elif args.integration_only:
        test_args.extend(["-m", "integration"])
    
    if args.verbose:
        test_args.append("-v")
    
    print("🧪 RTradez Test Coverage Analysis")
    print("="*50)
    
    # Run tests with coverage
    coverage_result = run_tests_with_coverage(test_args)
    
    if not coverage_result["success"]:
        print("❌ Tests failed!")
        if args.fail_under:
            sys.exit(1)
    
    # Generate coverage summary
    summary = generate_coverage_summary()
    
    if summary:
        save_coverage_report(summary, args.output)
        print_coverage_summary(summary)
        
        # Check threshold
        line_coverage = summary.get("overall", {}).get("line_rate", 0)
        meets_threshold = check_coverage_threshold(line_coverage, args.threshold)
        
        if args.fail_under and not meets_threshold:
            sys.exit(1)
    
    # Show HTML report location
    html_dir = Path("htmlcov")
    if html_dir.exists():
        print(f"\n📊 HTML coverage report available at: {html_dir.absolute()}/index.html")
    
    print("\n✅ Coverage analysis complete!")


if __name__ == "__main__":
    main()