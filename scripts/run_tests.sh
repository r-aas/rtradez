#!/bin/bash
# RTradez Test Runner Script
# 
# This script provides convenient commands for running different types of tests
# with coverage reporting and analysis.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
COVERAGE_THRESHOLD=80
VERBOSE=false
FAIL_UNDER=false
TEST_TYPE="all"

# Help function
show_help() {
    echo "RTradez Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS] [TEST_TYPE]"
    echo ""
    echo "TEST_TYPE:"
    echo "  all           Run all tests (default)"
    echo "  unit          Run only unit tests"
    echo "  integration   Run only integration tests"
    echo "  coverage      Run tests with detailed coverage analysis"
    echo "  quick         Run fast tests only (exclude slow tests)"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help              Show this help message"
    echo "  -v, --verbose           Verbose output"
    echo "  -t, --threshold NUM     Coverage threshold (default: 80)"
    echo "  -f, --fail-under        Fail if coverage below threshold"
    echo "  --html                  Generate HTML coverage report"
    echo "  --xml                   Generate XML coverage report"
    echo "  --json                  Generate JSON coverage report"
    echo "  --clean                 Clean previous coverage data"
    echo ""
    echo "Examples:"
    echo "  $0 unit                 # Run unit tests"
    echo "  $0 coverage -f -t 85    # Run coverage with 85% threshold"
    echo "  $0 --verbose all        # Run all tests with verbose output"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--threshold)
            COVERAGE_THRESHOLD="$2"
            shift 2
            ;;
        -f|--fail-under)
            FAIL_UNDER=true
            shift
            ;;
        --html)
            GENERATE_HTML=true
            shift
            ;;
        --xml)
            GENERATE_XML=true
            shift
            ;;
        --json)
            GENERATE_JSON=true
            shift
            ;;
        --clean)
            CLEAN_COVERAGE=true
            shift
            ;;
        unit|integration|all|coverage|quick)
            TEST_TYPE="$1"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to clean coverage data
clean_coverage() {
    print_status $BLUE "🧹 Cleaning previous coverage data..."
    rm -rf htmlcov/
    rm -f coverage.xml
    rm -f coverage_report.json
    rm -f .coverage
}

# Function to run tests
run_tests() {
    local test_args=()
    
    # Add verbose flag
    if [[ "$VERBOSE" == "true" ]]; then
        test_args+=("-v")
    fi
    
    # Add test type specific arguments
    case $TEST_TYPE in
        unit)
            test_args+=("-m" "unit")
            print_status $BLUE "🧪 Running unit tests..."
            ;;
        integration)
            test_args+=("-m" "integration")
            print_status $BLUE "🔗 Running integration tests..."
            ;;
        quick)
            test_args+=("-m" "not slow")
            print_status $BLUE "⚡ Running quick tests..."
            ;;
        coverage)
            test_args+=("--cov=src/rtradez" "--cov-report=term-missing")
            if [[ "$GENERATE_HTML" == "true" || "$TEST_TYPE" == "coverage" ]]; then
                test_args+=("--cov-report=html")
            fi
            if [[ "$GENERATE_XML" == "true" || "$TEST_TYPE" == "coverage" ]]; then
                test_args+=("--cov-report=xml")
            fi
            test_args+=("--cov-fail-under=$COVERAGE_THRESHOLD")
            print_status $BLUE "📊 Running tests with coverage analysis..."
            ;;
        all)
            print_status $BLUE "🚀 Running all tests..."
            ;;
    esac
    
    # Run tests
    /Users/r/.local/bin/uv run python -m pytest "${test_args[@]}"
    local exit_code=$?
    
    return $exit_code
}

# Function to generate coverage report
generate_coverage_report() {
    if [[ "$TEST_TYPE" == "coverage" || "$GENERATE_JSON" == "true" ]]; then
        print_status $BLUE "📋 Generating detailed coverage report..."
        /Users/r/.local/bin/uv run python scripts/test_coverage.py --output coverage_report.json
    fi
}

# Function to display results
display_results() {
    local exit_code=$1
    
    if [[ $exit_code -eq 0 ]]; then
        print_status $GREEN "✅ Tests completed successfully!"
        
        # Show coverage summary if available
        if [[ -f "coverage.xml" ]]; then
            print_status $BLUE "📊 Coverage Summary:"
            /Users/r/.local/bin/uv run python -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('coverage.xml')
    root = tree.getroot()
    line_rate = float(root.attrib.get('line-rate', 0)) * 100
    branch_rate = float(root.attrib.get('branch-rate', 0)) * 100
    print(f'  Line Coverage: {line_rate:.2f}%')
    print(f'  Branch Coverage: {branch_rate:.2f}%')
except Exception as e:
    print(f'  Could not parse coverage: {e}')
"
        fi
        
        # Show HTML report location
        if [[ -d "htmlcov" ]]; then
            print_status $YELLOW "📊 HTML coverage report: file://$(pwd)/htmlcov/index.html"
        fi
        
    else
        print_status $RED "❌ Tests failed!"
        
        if [[ "$FAIL_UNDER" == "true" ]]; then
            print_status $RED "💥 Exiting due to --fail-under flag"
            exit $exit_code
        fi
    fi
}

# Main execution
main() {
    print_status $BLUE "🧪 RTradez Test Runner"
    print_status $BLUE "======================"
    print_status $BLUE "Test Type: $TEST_TYPE"
    print_status $BLUE "Coverage Threshold: $COVERAGE_THRESHOLD%"
    print_status $BLUE "Verbose: $VERBOSE"
    print_status $BLUE "Fail Under: $FAIL_UNDER"
    echo ""
    
    # Clean coverage data if requested
    if [[ "$CLEAN_COVERAGE" == "true" ]]; then
        clean_coverage
    fi
    
    # Run tests
    run_tests
    local test_exit_code=$?
    
    # Generate additional reports
    generate_coverage_report
    
    # Display results
    display_results $test_exit_code
    
    # Return original exit code
    exit $test_exit_code
}

# Run main function
main