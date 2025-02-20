import os
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from detective import snapshot

from .fixtures_data import Cat, CocoCat, CocoDataclass, CocoProto
from .utils import are_snapshots_equal, get_debug_file, setup_debug_dir

# Test data for nested fields test
COCO_DATA = {
    "name": "Coco",
    "color": "calico",
    "foods": ["sushi", "salmon", "tuna"],
    "activities": [
        {"name": "sunbathing", "cuteness": "purrfectly_toasty"},
        {"name": "brushing", "adorableness": "melts_like_butter"},
    ],
}


class TestSnapshot:
    def setup_method(self):
        """Setup before each test."""
        setup_debug_dir()
        os.environ["DEBUG"] = "true"

    def test_debug_mode_off(self):
        """Test that no output is generated when debug mode is off."""
        os.environ["DEBUG"] = "false"

        @snapshot()
        def simple_function(x):
            return x * 2

        result = simple_function(5)
        assert result == 10

        debug_dir = os.path.join(os.getcwd(), "debug_snapshots")
        debug_files = [f for f in os.listdir(debug_dir) if f.endswith(".json")]
        assert (
            len(debug_files) == 0
        ), "No debug files should be created when debug is off"

    @patch("detective.snapshot.uuid.uuid4")
    def test_dataclass_serialization(self, mock_uuid):
        """Test serialization of dataclass objects."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def process_cat(cat: Cat) -> str:
            return f"{cat.name} likes {cat.foods[0]}"

        result = process_cat(CocoDataclass)
        assert result == "Coco likes sushi"

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "process_cat",
            "INPUTS": {
                "cat": {
                    "name": "Coco",
                    "color": "calico",
                    "foods": ["sushi", "salmon", "tuna"],
                    "activities": [
                        {"name": "sunbathing", "cuteness": "purrfectly_toasty"},
                        {"name": "brushing", "adorableness": "melts_like_butter"},
                    ],
                }
            },
            "OUTPUT": "Coco likes sushi",
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_protobuf_serialization(self, mock_uuid):
        """Test serialization of protobuf objects."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def color(cat_proto: Any) -> str:
            return cat_proto.color

        assert color(CocoProto) == "calico"

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "color",
            "INPUTS": {"cat_proto": CocoCat},
            "OUTPUT": "calico",
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_no_inputs_outputs(self, mock_uuid):
        """Test capturing a function with no inputs or outputs."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def func() -> None:
            pass

        func()

        # Get the actual output
        _, actual_data = get_debug_file(mock_uuid_str)

        # Create expected output
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": {},  # Empty dict since no inputs
            "OUTPUT": None,  # None since no return value
        }

        assert are_snapshots_equal(actual_data, expected_data)
