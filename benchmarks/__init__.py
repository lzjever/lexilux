"""Lexilux performance benchmarks.

This module contains performance benchmark tests for measuring and comparing
the performance characteristics of various Lexilux components, including:

- Connection pooling performance
- Request throughput
- Resource utilization

To run benchmarks:
    pytest benchmarks/ -v -m benchmark

Note: Most benchmarks are skipped by default as they require specific
configuration or external services. Remove the @pytest.mark.skipif decorators
to enable them.
"""
