#!/bin/bash
# Test runner script for Qdrant RAG MCP

set -e

echo "🧪 Qdrant RAG MCP Test Runner"
echo "============================"

# Function to run specific test category
run_category() {
    local category=$1
    echo
    echo "Running $category tests..."
    uv run pytest tests/$category/ -v
}

# Function to run all tests
run_all() {
    echo
    echo "Running all tests..."
    uv run pytest tests/ -v
}

# Function to run with coverage
run_coverage() {
    echo
    echo "Running tests with coverage..."
    uv run pytest tests/ --cov=src --cov-report=html --cov-report=term
    echo "Coverage report generated in htmlcov/index.html"
}

# Main menu
case ${1:-help} in
    unit)
        run_category unit
        ;;
    integration)
        run_category integration
        ;;
    performance)
        run_category performance
        ;;
    debug)
        echo "Debug scripts are not pytest tests. Run them directly with:"
        echo "  uv run python tests/debug/<script_name>.py"
        ;;
    all)
        run_all
        ;;
    coverage)
        run_coverage
        ;;
    quick)
        echo
        echo "Running quick unit tests only..."
        uv run pytest tests/unit/ -v -m "not slow"
        ;;
    *)
        echo "Usage: $0 [unit|integration|performance|all|coverage|quick]"
        echo
        echo "Options:"
        echo "  unit         - Run unit tests"
        echo "  integration  - Run integration tests"
        echo "  performance  - Run performance tests"
        echo "  all          - Run all tests"
        echo "  coverage     - Run all tests with coverage report"
        echo "  quick        - Run quick unit tests only"
        echo
        echo "Examples:"
        echo "  $0 unit"
        echo "  $0 coverage"
        ;;
esac