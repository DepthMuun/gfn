"""
GFN Test Suite Organization
============================

This directory contains comprehensive tests for GFN realizations.
Tests are organized into three categories:

Directory Structure
-------------------

health/         - Core functionality tests (unit tests, smoke tests)
research/       - Deep analysis and investigation tests
benchmarks/     - Performance and convergence benchmarks

Usage
-----
Run all tests:
    pytest tests/ -v

Run specific category:
    pytest tests/gssm/health/ -v      # Unit tests
    pytest tests/gssm/research/ -v    # Research tests
    pytest tests/gssm/benchmarks/ -v  # Benchmarks

ISN Tests
---------
The same structure applies for ISN tests under tests/isn/.
"""
