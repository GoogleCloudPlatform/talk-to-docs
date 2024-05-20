"""This module provides tests for measure_utils.py"""

from unittest.mock import patch
from gen_ai.common.measure_utils import trace_on
from gen_ai.common.ioc_container import Container


def test_trace_on_logging():
    """Test the trace_on decorator for proper logging without execution time measurement."""
    with patch("gen_ai.common.ioc_container.Container.logger") as mock_logger:
        Container.config["print_system_metrics"] = True

        @trace_on("Testing function execution")
        def test_function():
            return "Function executed"

        result = test_function()

        assert result == "Function executed"

        mock_logger().info.assert_called_with(msg="Testing function execution")


def test_trace_on_with_time_measurement():
    """Test the trace_on decorator for correct execution time measurement and logging."""
    with patch("gen_ai.common.ioc_container.Container.logger") as mock_logger:
        Container.config["print_system_metrics"] = True

        @trace_on("Timing function", measure_time=True)
        def test_function():
            return "Function executed"

        # Call the decorated function
        result = test_function()

        # Check that the function returns the correct result
        assert result == "Function executed"

        # Assert that the log message contains the expected text
        assert mock_logger().info.call_args.startswith("Timing function took")
        # Additionally, you could use a regex to assert more precisely that it logs a time.
