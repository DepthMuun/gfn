#!/bin/bash
# GFN Test Runner for Linux/Mac
# Usage: ./run_tests.sh [test_file]

TEST_FILE="${1:-tests/unit/}"

echo "Running GFN Tests..."
echo "Test target: $TEST_FILE"
echo ""

# Set PYTHONPATH to project root
export PYTHONPATH="$(dirname "$0")"

# Run pytest
python -m pytest "$TEST_FILE" -v --tb=short

echo ""
echo "Tests completed!"
