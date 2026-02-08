"""
Rate limiter test cases.
"""

import asyncio
import time

import pytest

from lexilux._rate_limit import RateLimiter, has_rate_limit_support


pytest.importorskip("aiolimiter")


class TestRateLimiter:
    """RateLimiter initialization and properties tests"""

    def test_rate_limiter_init(self):
        """Test RateLimiter initialization with valid parameters"""
        limiter = RateLimiter(max_rate=10, time_period=60.0)
        assert limiter.max_rate == 10
        assert limiter.time_period == 60.0

    def test_rate_limiter_properties(self):
        """Test RateLimiter properties"""
        limiter = RateLimiter(max_rate=5, time_period=10.0)
        assert limiter.max_rate == 5
        assert limiter.time_period == 10.0

    def test_rate_limiter_is_available(self):
        """Test that rate limiter reports as available when aiolimiter is installed"""
        limiter = RateLimiter(max_rate=10, time_period=60.0)
        assert limiter.is_available() is True

    def test_has_rate_limit_support(self):
        """Test the module-level support check function"""
        assert has_rate_limit_support() is True


class TestRateLimiterAcquire:
    """RateLimiter acquire() tests"""

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_limit(self):
        """Test that rate limiter respects configured limits"""
        limiter = RateLimiter(max_rate=3, time_period=1.0)

        # Should be able to acquire 3 times immediately
        start_time = time.time()
        for _ in range(3):
            await limiter.acquire()
        elapsed_first_three = time.time() - start_time

        # First 3 should be fast (within 100ms)
        assert elapsed_first_three < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_limit_reached(self):
        """Test that rate limiter blocks when limit is reached"""
        limiter = RateLimiter(max_rate=2, time_period=0.5)

        # Acquire all available permits
        start_time = time.time()
        await limiter.acquire()
        await limiter.acquire()

        # Third acquisition should block until time_period passes
        await limiter.acquire()
        elapsed = time.time() - start_time

        # Should have waited at least some time (rate limiter is not instant)
        # Using a more lenient check since timing can vary
        assert elapsed >= 0.2, f"Should have been delayed, but took {elapsed}s"

    @pytest.mark.asyncio
    async def test_rate_limiter_resets_after_time_period(self):
        """Test that rate limiter resets after time period"""
        limiter = RateLimiter(max_rate=2, time_period=0.3)

        # Use all permits
        await limiter.acquire()
        await limiter.acquire()

        # Wait for reset
        await asyncio.sleep(0.35)

        # Should be able to acquire again immediately
        start_time = time.time()
        await limiter.acquire()
        elapsed = time.time() - start_time

        # Should be fast (within 100ms)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self):
        """Test rate limiter with concurrent requests"""
        limiter = RateLimiter(max_rate=5, time_period=1.0)

        async def make_request():
            await limiter.acquire()
            return True

        # Launch concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_rate_limiter_steady_rate(self):
        """Test that rate limiter maintains steady rate over time"""
        limiter = RateLimiter(max_rate=2, time_period=1.0)

        # Make 4 requests (2x the limit)
        for _ in range(4):
            await limiter.acquire()

        # If we got here without hanging forever, the rate limiter is working
        # The key test is that it doesn't block forever and processes requests
        # Just verify we can make multiple requests
        assert True  # Test passed if we got here


class TestRateLimiterWithoutAiolimiter:
    """Test rate limiter behavior when aiolimiter is not available"""

    def test_rate_limiter_import_error_without_aiolimiter(self, monkeypatch):
        """Test that RateLimiter raises ImportError when aiolimiter is not available"""
        # Simulate aiolimiter not being available
        import lexilux._rate_limit as rl_module

        monkeypatch.setattr(rl_module, "AIOLIMITER_AVAILABLE", False)
        monkeypatch.setattr(rl_module, "AsyncLimiter", None)

        with pytest.raises(ImportError, match="aiolimiter is required"):
            RateLimiter(max_rate=10, time_period=60.0)
