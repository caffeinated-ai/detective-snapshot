import os
import uuid
from unittest.mock import patch

import pytest

from detective import snapshot
from detective.snapshot import inner_calls_var, session_id_var
from detective.tests.fixtures_data import Cat, CocoDataclass
from detective.tests.utils import are_snapshots_equal, get_debug_file, setup_debug_dir


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
    @patch("detective.snapshot.uuid.uuid4")
    def test_instance_method_exception(self, mock_uuid):
        """Test exception handling for instance methods."""
        mock_uuid_str = "12345678-1234-5678-1234-567812345678"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        cat_handler = ErrorProneCat(CocoDataclass)
        with pytest.raises(ValueError, match="Instance method error"):
            cat_handler.instance_error()

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "instance_error",
            "INPUTS": {"self": {"cat": CocoDataclass.to_dict()}},
            "ERROR": {"type": "ValueError", "message": "Instance method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_class_method_exception(self, mock_uuid):
        """Test exception handling for class methods."""
        mock_uuid_str = "23456789-2345-6789-2345-678923456789"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        with pytest.raises(TypeError, match="Class method error"):
            ErrorProneCat.class_error()

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "class_error",
            "INPUTS": {"cls": {"default_cat": CocoDataclass.to_dict()}},
            "ERROR": {"type": "TypeError", "message": "Class method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_static_method_exception(self, mock_uuid):
        """Test exception handling for static methods."""
        mock_uuid_str = "34567890-3456-7890-3456-789034567890"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        with pytest.raises(RuntimeError, match="Static method error"):
            ErrorProneCat.static_error()

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "static_error",
            "INPUTS": {},
            "ERROR": {"type": "RuntimeError", "message": "Static method error"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_nested_function_exception(self, mock_uuid):
        """Test exception handling for nested function calls."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

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

        _, actual_data = get_debug_file(mock_uuid_str)
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
