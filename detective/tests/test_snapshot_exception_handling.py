import os
import uuid
from unittest.mock import patch

import pytest

from detective import snapshot
from detective.snapshot import inner_calls_var, session_id_var
from detective.tests.fixtures_data import Cat, CocoDataclass
from detective.tests.utils import (
    are_snapshots_equal,
    get_debug_file,
    get_test_hash,
    setup_debug_dir,
)


@pytest.fixture(autouse=True)
def setup_module():
    """Setup for each test."""
    os.environ["DEBUG"] = "true"
    setup_debug_dir()
    yield
    session_id_var.set(None)
    inner_calls_var.set([])


class ErrorProneCat:
    default_cat = CocoDataclass

    def __init__(self, cat: Cat):
        self.cat = cat

    @snapshot()
    def instance_error(self) -> None:
        raise ValueError("Instance method error")

    @classmethod
    @snapshot()
    def class_error(cls) -> None:
        raise TypeError("Class method error")

    @staticmethod
    @snapshot()
    def static_error() -> None:
        raise RuntimeError("Static method error")


class TestExceptionHandling:
    @patch("detective.snapshot._generate_short_hash")
    def test_instance_method_exception(self, mock_hash):
        """Test exception handling for instance methods."""
        mock_hash.return_value = get_test_hash()

        cat_handler = ErrorProneCat(CocoDataclass)
        with pytest.raises(ValueError, match="Instance method error"):
            cat_handler.instance_error()

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "instance_error",
            "INPUTS": {},
            "ERROR": {"type": "ValueError", "message": "Instance method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_class_method_exception(self, mock_hash):
        """Test exception handling for class methods."""
        mock_hash.return_value = get_test_hash("second")

        with pytest.raises(TypeError, match="Class method error"):
            ErrorProneCat.class_error()

        _, actual_data = get_debug_file(get_test_hash("second"))
        expected_data = {
            "FUNCTION": "class_error",
            "INPUTS": {},
            "ERROR": {"type": "TypeError", "message": "Class method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_static_method_exception(self, mock_hash):
        """Test exception handling for static methods."""
        mock_hash.return_value = get_test_hash("third")

        with pytest.raises(RuntimeError, match="Static method error"):
            ErrorProneCat.static_error()

        _, actual_data = get_debug_file(get_test_hash("third"))
        expected_data = {
            "FUNCTION": "static_error",
            "INPUTS": {},
            "ERROR": {"type": "RuntimeError", "message": "Static method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_nested_function_exception(self, mock_hash):
        """Test exception handling for nested function calls."""
        mock_hash.return_value = get_test_hash("fourth")

        @snapshot()
        def outer():
            return inner()

        @snapshot()
        def inner():
            return problematic()

        @snapshot()
        def problematic():
            raise ValueError("Something went wrong!")

        with pytest.raises(ValueError) as exc_info:
            outer()
        assert str(exc_info.value) == "Something went wrong!"

        _, actual_data = get_debug_file(get_test_hash("fourth"))
        expected_data = {
            "FUNCTION": "outer",
            "INPUTS": {},
            "ERROR": {"type": "ValueError", "message": "Something went wrong!"},
            "CALLS": [
                {
                    "FUNCTION": "inner",
                    "INPUTS": {},
                    "ERROR": {"type": "ValueError", "message": "Something went wrong!"},
                    "CALLS": [
                        {
                            "FUNCTION": "problematic",
                            "INPUTS": {},
                            "ERROR": {
                                "type": "ValueError",
                                "message": "Something went wrong!",
                            },
                        }
                    ],
                }
            ],
        }
        assert are_snapshots_equal(actual_data, expected_data)


class TestExceptionHandlingWithFieldsSelection:
    @patch("detective.snapshot._generate_short_hash")
    def test_instance_method_exception_with_field_selection(self, mock_hash):
        """Test exception handling for instance methods with field selection."""
        mock_hash.return_value = get_test_hash()

        class ErrorProneCatWithFields(ErrorProneCat):
            @snapshot(input_fields=["self.cat"], include_implicit=True)
            def instance_error(self) -> None:
                raise ValueError("Instance method error")

        cat_handler = ErrorProneCatWithFields(CocoDataclass)
        with pytest.raises(ValueError, match="Instance method error"):
            cat_handler.instance_error()

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "instance_error",
            "INPUTS": {"self": {"cat": CocoDataclass.to_dict()}},
            "ERROR": {"type": "ValueError", "message": "Instance method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_class_method_exception_with_field_selection(self, mock_hash):
        """Test exception handling for class methods with field selection."""
        mock_hash.return_value = get_test_hash("second")

        class ErrorProneCatWithFields(ErrorProneCat):
            @classmethod
            @snapshot(input_fields=["cls.default_cat"], include_implicit=True)
            def class_error(cls) -> None:
                raise TypeError("Class method error")

        with pytest.raises(TypeError, match="Class method error"):
            ErrorProneCatWithFields.class_error()

        _, actual_data = get_debug_file(get_test_hash("second"))
        expected_data = {
            "FUNCTION": "class_error",
            "INPUTS": {"cls": {}},
            "ERROR": {"type": "TypeError", "message": "Class method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_static_method_exception_with_field_selection(self, mock_hash):
        """Test exception handling for static methods with field selection."""
        mock_hash.return_value = get_test_hash("third")

        class ErrorProneCatWithFields(ErrorProneCat):
            @staticmethod
            @snapshot(input_fields=["message"])
            def static_error_with_args(message: str) -> None:
                raise RuntimeError(message)

        with pytest.raises(RuntimeError, match="Custom error message"):
            ErrorProneCatWithFields.static_error_with_args("Custom error message")

        _, actual_data = get_debug_file(get_test_hash("third"))
        expected_data = {
            "FUNCTION": "static_error_with_args",
            "INPUTS": {"message": "Custom error message"},
            "ERROR": {"type": "RuntimeError", "message": "Custom error message"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_nested_function_exception_with_field_selection(self, mock_hash):
        """Test exception handling for nested function calls with field selection."""
        mock_hash.return_value = get_test_hash("fourth")

        @snapshot(input_fields=["x", "y"])
        def outer(x, y, z):
            return inner(x + y, z)

        @snapshot(input_fields=["a"])
        def inner(a, b):
            return problematic(a, b)

        @snapshot(input_fields=["error_message"])
        def problematic(c, error_message):
            raise ValueError(error_message)

        with pytest.raises(ValueError) as exc_info:
            outer(10, 20, "Error in nested call")
        assert str(exc_info.value) == "Error in nested call"

        _, actual_data = get_debug_file(get_test_hash("fourth"))
        expected_data = {
            "FUNCTION": "outer",
            "INPUTS": {"x": 10, "y": 20},
            "ERROR": {"type": "ValueError", "message": "Error in nested call"},
            "CALLS": [
                {
                    "FUNCTION": "inner",
                    "INPUTS": {"a": 30},
                    "ERROR": {"type": "ValueError", "message": "Error in nested call"},
                    "CALLS": [
                        {
                            "FUNCTION": "problematic",
                            "INPUTS": {"error_message": "Error in nested call"},
                            "ERROR": {
                                "type": "ValueError",
                                "message": "Error in nested call",
                            },
                        }
                    ],
                }
            ],
        }
        assert are_snapshots_equal(actual_data, expected_data)
