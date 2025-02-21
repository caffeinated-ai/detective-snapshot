import json
import os
import time
import uuid
from typing import Any, Dict
from unittest.mock import patch

import pytest

from detective import snapshot

from .fixtures_data import Cat, CocoCat, CocoDataclass, CocoProto
from .utils import (
    are_snapshots_equal,
    get_debug_file,
    get_test_hash,
    mock_hash_sequence,
    setup_debug_dir,
)

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

    @patch("detective.snapshot._generate_short_hash")
    def test_debug_mode_off(self, mock_hash):
        """Test that no output is generated when debug mode is off."""
        os.environ["DEBUG"] = "0"
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def simple_function(x):
            return x * 2

        result = simple_function(5)
        assert result == 10

        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        assert not os.path.exists(debug_dir) or len(os.listdir(debug_dir)) == 0

    @patch("detective.snapshot._generate_short_hash")
    def test_dataclass_serialization(self, mock_hash):
        """Test serialization of dataclass objects."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def process_cat(cat: Cat) -> str:
            return f"{cat.name} likes {cat.foods[0]}"

        result = process_cat(CocoDataclass)
        assert result == "Coco likes sushi"

        _, actual_data = get_debug_file(get_test_hash())
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

    @patch("detective.snapshot._generate_short_hash")
    def test_protobuf_serialization(self, mock_hash):
        """Test serialization of protobuf objects."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def color(cat_proto: Any) -> str:
            return cat_proto.color

        assert color(CocoProto) == "calico"

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "color",
            "INPUTS": {"cat_proto": CocoCat},
            "OUTPUT": "calico",
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_no_inputs_outputs(self, mock_hash):
        """Test capturing a function with no inputs or outputs."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def func() -> None:
            pass

        func()

        # Get the actual output
        _, actual_data = get_debug_file(get_test_hash())

        # Create expected output
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": {},  # Empty dict since no inputs
            "OUTPUT": None,  # None since no return value
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_multiple_function_calls(self, mock_hash):
        """Test that multiple calls to a snapshotted function create separate debug files."""
        # Set up mock hashes for the two calls
        mock_hash.side_effect = mock_hash_sequence(2)

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
        _, actual_data1 = get_debug_file(get_test_hash())
        _, actual_data2 = get_debug_file(get_test_hash("second"))

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

        # Verify that two debug files were created
        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        debug_files = sorted(
            [f for f in os.listdir(debug_dir) if f.startswith("greet_")]
        )
        assert len(debug_files) == 2

    @patch("detective.snapshot._generate_short_hash")
    def test_method_with_field_selection(self, mock_hash):
        """Test capturing a method call with field selection."""
        mock_hash.return_value = get_test_hash()

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

        _, actual_data = get_debug_file(get_test_hash())
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

    @patch("detective.snapshot._generate_short_hash")
    def test_classmethod_with_field_selection(self, mock_hash):
        """Test capturing a class method call with field selection."""
        mock_hash.return_value = get_test_hash()

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

        _, actual_data = get_debug_file(get_test_hash())
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

    @patch("detective.snapshot._generate_short_hash")
    def test_staticmethod_with_field_selection(self, mock_hash):
        """Test capturing a static method call with field selection."""
        mock_hash.return_value = get_test_hash()

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

        _, actual_data = get_debug_file(get_test_hash())
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

    @patch("detective.snapshot._generate_short_hash")
    @patch("time.localtime")
    def test_snapshot_filename_generation(self, mock_localtime, mock_hash):
        """Test that snapshot filenames are generated correctly."""
        # Mock the hash
        mock_hash.return_value = "abc123d"

        # Mock two different timestamps
        first_time = time.struct_time(
            (2024, 3, 5, 10, 30, 0, 0, 0, 0)
        )  # March 5, 10:30:00
        second_time = time.struct_time(
            (2024, 3, 5, 10, 30, 5, 0, 0, 0)
        )  # March 5, 10:30:05

        mock_localtime.side_effect = [first_time, second_time]

        @snapshot()
        def my_function(x: int) -> int:
            return x * 2

        # Make two calls to generate two files
        result1 = my_function(5)
        result2 = my_function(10)

        assert result1 == 10
        assert result2 == 20

        # Check the files in the snapshots directory
        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        assert os.path.exists(debug_dir), "Snapshot directory not created"

        files = sorted(os.listdir(debug_dir))
        assert len(files) == 2, f"Expected 2 files, found: {files}"

        # Verify filenames follow the pattern: {function_name}_{timestamp}_{hash}.json
        expected_files = [
            "my_function_0305103000_abc123d.json",  # March 5, 10:30:00
            "my_function_0305103005_abc123d.json",  # March 5, 10:30:05
        ]
        assert files == expected_files, f"Unexpected files: {files}"

        # Verify file contents
        for filename, expected_input, expected_output in [
            (expected_files[0], 5, 10),
            (expected_files[1], 10, 20),
        ]:
            with open(os.path.join(debug_dir, filename), encoding="utf-8") as f:
                data = json.load(f)
                assert data["FUNCTION"] == "my_function"
                assert data["INPUTS"] == {"x": expected_input}
                assert data["OUTPUT"] == expected_output

    @patch("detective.snapshot._generate_short_hash")
    def test_detective_env_var(self, mock_hash):
        """Test that DETECTIVE environment variable enables debugging."""
        mock_hash.return_value = get_test_hash()

        # Set DETECTIVE=true and unset DEBUG
        os.environ["DETECTIVE"] = "true"
        os.environ.pop("DEBUG", None)

        @snapshot()
        def simple_function(x):
            return x * 2

        result = simple_function(5)
        assert result == 10

        # Verify debug file was created
        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "simple_function",
            "INPUTS": {"x": 5},
            "OUTPUT": 10,
        }
        assert are_snapshots_equal(actual_data, expected_data)

        # Test with DETECTIVE=1
        os.environ["DETECTIVE"] = "1"
        result = simple_function(7)
        assert result == 14

        # Test with invalid value
        os.environ["DETECTIVE"] = "yes"
        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        file_count_before = len(os.listdir(debug_dir))
        result = simple_function(9)
        assert result == 18
        assert (
            len(os.listdir(debug_dir)) == file_count_before
        ), "No new files should be created"

    @patch("detective.snapshot._generate_short_hash")
    @patch("time.localtime")
    def test_detective_env_end_to_end(self, mock_localtime, mock_hash):
        """End-to-end test using DETECTIVE environment variable."""
        # Set up mocks
        mock_hash.return_value = get_test_hash()
        mock_localtime.return_value = time.struct_time((2024, 3, 5, 10, 30, 0, 0, 0, 0))

        # Set DETECTIVE=true and unset DEBUG
        os.environ["DETECTIVE"] = "true"
        os.environ.pop("DEBUG", None)

        class CatBehavior:
            def __init__(self, cat: Cat):
                self.cat = cat

            @snapshot(
                input_fields=["self.cat.name", "self.cat.foods"],
                output_fields=["favorite_food"],
            )
            def get_favorite_food(self) -> Dict[str, Any]:
                return {
                    "favorite_food": self.cat.foods[0],
                    "color": self.cat.color,  # This should not be captured
                    "timestamp": "2024-01-01",  # This should not be captured
                }

        # Create and use the class
        behavior = CatBehavior(CocoDataclass)
        result = behavior.get_favorite_food()

        # Verify the actual function result is complete
        assert result == {
            "favorite_food": "sushi",
            "color": "calico",
            "timestamp": "2024-01-01",
        }

        # Verify the debug snapshot was created with selected fields
        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "get_favorite_food",
            "INPUTS": {
                "self": {"cat": {"name": "Coco", "foods": ["sushi", "salmon", "tuna"]}}
            },
            "OUTPUT": {"favorite_food": "sushi"},
        }
        assert are_snapshots_equal(actual_data, expected_data)

        # Verify the snapshot file was created with correct name format
        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        files = sorted(
            [f for f in os.listdir(debug_dir) if f.startswith("get_favorite_food_")]
        )
        assert len(files) == 1, f"Expected 1 file, found: {files}"
        assert files[0] == "get_favorite_food_0305103000_abc123d.json"

        # Test that snapshots stop when DETECTIVE is set to invalid value
        os.environ["DETECTIVE"] = "no"
        file_count_before = len(files)

        another_result = behavior.get_favorite_food()
        assert another_result == result  # Function still works

        files_after = [
            f for f in os.listdir(debug_dir) if f.startswith("get_favorite_food_")
        ]
        assert len(files_after) == file_count_before  # No new files created
