# ============================================================
# tests/unit/test_circuit_breaker.py
# Unit tests — CircuitBreaker for GraphRAG pipeline
# ============================================================

import time
import pytest
from core.graphrag.pipeline import CircuitBreaker, CircuitState


class TestCircuitBreaker:

    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb._state == CircuitState.CLOSED
        assert cb.is_open() is False

    def test_stays_closed_on_successes(self):
        cb = CircuitBreaker()
        for _ in range(20):
            cb.record_success()
        assert cb._state == CircuitState.CLOSED

    def test_trips_on_high_error_rate(self):
        cb = CircuitBreaker(error_threshold_pct=30.0, window_seconds=60.0)
        # 4 successes + 6 failures = 60% error rate > 30% threshold
        for _ in range(4):
            cb.record_success()
        for _ in range(6):
            cb.record_failure()
        assert cb._state == CircuitState.OPEN
        assert cb.is_open() is True

    def test_does_not_trip_below_threshold(self):
        cb = CircuitBreaker(error_threshold_pct=30.0)
        # 8 successes + 2 failures = 20% error rate < 30%
        for _ in range(8):
            cb.record_success()
        for _ in range(2):
            cb.record_failure()
        assert cb._state == CircuitState.CLOSED

    def test_does_not_trip_with_insufficient_data(self):
        cb = CircuitBreaker(error_threshold_pct=30.0)
        # Only 4 calls — below minimum of 5
        for _ in range(3):
            cb.record_failure()
        cb.record_success()
        assert cb._state == CircuitState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        cb = CircuitBreaker(
            error_threshold_pct=30.0,
            recovery_timeout_seconds=0.05,  # 50ms for test speed
        )
        for _ in range(4):
            cb.record_success()
        for _ in range(6):
            cb.record_failure()

        assert cb._state == CircuitState.OPEN
        time.sleep(0.1)  # Wait past recovery timeout

        # is_open() triggers the half-open transition
        result = cb.is_open()
        assert result is False
        assert cb._state == CircuitState.HALF_OPEN

    def test_closes_after_successful_half_open_calls(self):
        cb = CircuitBreaker(
            error_threshold_pct=30.0,
            recovery_timeout_seconds=0.05,
            half_open_max_calls=3,
        )
        for _ in range(4):
            cb.record_success()
        for _ in range(6):
            cb.record_failure()

        time.sleep(0.1)
        cb.is_open()  # Trigger half-open

        # 3 successes in half-open → close
        for _ in range(3):
            cb.record_success()

        assert cb._state == CircuitState.CLOSED

    def test_error_rate_window_prunes_old_events(self):
        cb = CircuitBreaker(error_threshold_pct=30.0, window_seconds=0.1,
                            recovery_timeout_seconds=0.05)
        for _ in range(6):
            cb.record_failure()
        assert cb._state == CircuitState.OPEN
        time.sleep(0.15)  # Let window expire AND recovery timeout pass
        # Trigger half-open via is_open()
        cb.is_open()
        # 3 successes in half-open → close
        for _ in range(3):
            cb.record_success()
        assert cb._state == CircuitState.CLOSED