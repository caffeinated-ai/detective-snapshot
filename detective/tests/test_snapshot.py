import os
import uuid
from typing import Any, Dict
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
        os.environ["DEBUG"] = "0"

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

    @patch("detective.snapshot.uuid.uuid4")
    def test_multiple_function_calls(self, mock_uuid):
        """Test that multiple calls to a snapshotted function create separate debug files."""
        # Set up mock UUIDs for the two calls
        mock_uuids = [
            "45678901-4567-8901-4567-890145678901",
            "56789012-5678-9012-5678-901256789012",
        ]
        mock_uuid.side_effect = [uuid.UUID(uid) for uid in mock_uuids]

        @snapshot()
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        # Call the function twice with different inputs
        result1 = greet("Alice")
        result2 = greet("Bob")

        # Verify function results
        assert result1 == "Hello, Alice!"
        assert result2 == "Hello, Bob!"

        # Check both debug files exist with correct content
        _, actual_data1 = get_debug_file(mock_uuids[0])
        _, actual_data2 = get_debug_file(mock_uuids[1])

        expected_data1 = {
            "FUNCTION": "greet",
            "INPUTS": {"name": "Alice"},
            "OUTPUT": "Hello, Alice!",
        }

        expected_data2 = {
            "FUNCTION": "greet",
            "INPUTS": {"name": "Bob"},
            "OUTPUT": "Hello, Bob!",
        }

        assert are_snapshots_equal(actual_data1, expected_data1)
        assert are_snapshots_equal(actual_data2, expected_data2)

        # Verify number of debug files
        debug_dir = os.path.join(os.getcwd(), "debug_snapshots")
        debug_files = [f for f in os.listdir(debug_dir) if f.endswith(".json")]
        assert len(debug_files) == 2, "Expected two debug files to be created"

    @patch("detective.snapshot.uuid.uuid4")
    def test_method_with_field_selection(self, mock_uuid):
        """Test capturing a method call with field selection."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        class CatBehavior:
            def __init__(self, cat: Cat):
                self.cat = cat

            @snapshot(
                input_fields=[
                    "self.cat.name",
                    "self.cat.foods[0]",
                    "meal_time.hour",
                    "meal_time.period",
                ],
                output_fields=["favorite_food", "time"],
            )
            def get_favorite_food(self, meal_time: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "name": self.cat.name,
                    "favorite_food": self.cat.foods[0],
                    "time": f"{meal_time['hour']} {meal_time['period']}",
                    "other_detail": "not captured",
                }

        behavior = CatBehavior(CocoDataclass)
        meal_time = {
            "hour": 6,
            "period": "PM",
            "timezone": "PST",  # This should not be captured
        }
        result = behavior.get_favorite_food(meal_time)

        assert result == {
            "name": "Coco",
            "favorite_food": "sushi",
            "time": "6 PM",
            "other_detail": "not captured",
        }

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "get_favorite_food",
            "INPUTS": {
                "self": {"cat": {"name": "Coco", "foods": ["sushi"]}},
                "meal_time": {"hour": 6, "period": "PM"},
            },
            "OUTPUT": {
                "favorite_food": "sushi",
                "time": "6 PM",
            },
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_classmethod_with_field_selection(self, mock_uuid):
        """Test capturing a class method call with field selection."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        class CatFactory:
            default_cat = CocoDataclass

            @classmethod
            @snapshot(
                input_fields=["cls.default_cat.name", "cls.default_cat.foods"],
                output_fields=["name", "foods"],
            )
            def create_default_cat(cls) -> Dict[str, Any]:
                return {
                    "name": cls.default_cat.name,
                    "foods": cls.default_cat.foods,
                    "color": cls.default_cat.color,
                    "internal_id": "123",
                }

        result = CatFactory.create_default_cat()

        assert result == {
            "name": "Coco",
            "foods": ["sushi", "salmon", "tuna"],
            "color": "calico",
            "internal_id": "123",
        }

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "create_default_cat",
            "INPUTS": {
                "cls": {
                    "default_cat": {
                        "name": "Coco",
                        "foods": ["sushi", "salmon", "tuna"],
                    }
                }
            },
            "OUTPUT": {
                "name": "Coco",
                "foods": ["sushi", "salmon", "tuna"],
            },
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_staticmethod_with_field_selection(self, mock_uuid):
        """Test capturing a static method call with field selection."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        class CatValidator:
            @staticmethod
            @snapshot(
                input_fields=["cat.name", "cat.foods"],
                output_fields=["is_valid", "food_count"],
            )
            def validate_cat(cat: Cat) -> Dict[str, Any]:
                return {
                    "is_valid": bool(cat.name and cat.foods),
                    "food_count": len(cat.foods),
                    "internal_check": "passed",
                    "timestamp": "2024-01-01",
                }

        result = CatValidator.validate_cat(CocoDataclass)

        assert result == {
            "is_valid": True,
            "food_count": 3,
            "internal_check": "passed",
            "timestamp": "2024-01-01",
        }

        _, actual_data = get_debug_file(mock_uuid_str)
        expected_data = {
            "FUNCTION": "validate_cat",
            "INPUTS": {
                "cat": {
                    "name": "Coco",
                    "foods": ["sushi", "salmon", "tuna"],
                }
            },
            "OUTPUT": {
                "is_valid": True,
                "food_count": 3,
            },
        }

        assert are_snapshots_equal(actual_data, expected_data)
