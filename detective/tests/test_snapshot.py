import functools
import json
import os
import time
import uuid
from typing import Any, Dict
from unittest.mock import patch

import pytest
import sympy as sp

from detective import snapshot
from detective.snapshot import CustomJSONEncoder

from .fixtures_data import Cat, CocoCat, CocoDataclass, CocoProto
from .utils import (are_snapshots_equal, get_debug_file, get_test_hash,
                    mock_hash_sequence, setup_debug_dir)

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

    @patch("detective.snapshot._generate_short_hash")
    def test_stacked_decorators(self, mock_hash):
        """Test that @snapshot works correctly when stacked with other decorators."""
        mock_hash.return_value = get_test_hash()

        def my_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Simple decorator that adds 1 to the result
                result = func(*args, **kwargs)
                return result + 1

            return wrapper

        class Calculator:
            @my_decorator
            @snapshot(input_fields=["x", "y"])
            def add(self, x: int, y: int) -> int:
                return x + y

            @snapshot(input_fields=["x", "y"])
            @my_decorator
            def subtract(self, x: int, y: int) -> int:
                return x - y

        calc = Calculator()

        # Test decorator before snapshot
        result1 = calc.add(5, 3)  # Should be (5 + 3) + 1 = 9
        assert result1 == 9

        _, actual_data1 = get_debug_file(get_test_hash())
        expected_data1 = {
            "FUNCTION": "add",
            "INPUTS": {"x": 5, "y": 3},
            "OUTPUT": 8,  # Snapshot captures the original return value before my_decorator
        }
        assert are_snapshots_equal(actual_data1, expected_data1)

        # Reset hash for second test
        mock_hash.return_value = get_test_hash("second")

        # Test snapshot before decorator
        result2 = calc.subtract(10, 4)  # Should be (10 - 4) + 1 = 7
        assert result2 == 7

        _, actual_data2 = get_debug_file(get_test_hash("second"))
        expected_data2 = {
            "FUNCTION": "subtract",
            "INPUTS": {"x": 10, "y": 4},
            "OUTPUT": 7,  # Snapshot captures the final return value after my_decorator
        }
        assert are_snapshots_equal(actual_data2, expected_data2)

    @patch("detective.snapshot._generate_short_hash")
    def test_sympy_symbol_serialization(self, mock_hash):
        """Test that SymPy Symbols are properly serialized."""
        mock_hash.return_value = get_test_hash()

        # Create a SymPy symbol
        x = sp.Symbol('x')
        y = sp.Symbol('y')

        # Create test data with symbols as both keys and values
        test_data = {
            x: "value_with_symbol_key",
            "normal_key": y,
            "expression": x + y,  # SymPy handles this correctly at runtime
            "nested": {
                y: [x, y, x + y],
                "symbols": {x: y}
            }
        }

        @snapshot()
        def process_symbols(data):
            return data

        result = process_symbols(test_data)

        # Verify the function returned the original data
        assert result == test_data

        # Get the debug file and verify serialization
        _, actual_data = get_debug_file(get_test_hash())

        # Convert the actual data to JSON and back to verify serialization
        json_str = json.dumps(actual_data, cls=CustomJSONEncoder)
        decoded = json.loads(json_str)

        # Verify the results
        assert "INPUTS" in decoded
        inputs_data = decoded["INPUTS"]["data"]
        assert "x" in inputs_data  # Symbol key should be converted to string
        assert isinstance(inputs_data["normal_key"], str)  # Symbol value should be converted to string
        assert isinstance(inputs_data["expression"], str)  # Expression should be converted to string
        assert "y" in inputs_data["nested"]  # Nested Symbol key should be converted
        assert isinstance(inputs_data["nested"]["symbols"]["x"], str)  # Deeply nested Symbol should be converted

    @patch("detective.snapshot._generate_short_hash")
    def test_snapshot_with_sympy(self, mock_hash):
        """Test the Snapshotter with SymPy objects."""
        mock_hash.return_value = get_test_hash()

        # Create a function that uses SymPy
        @snapshot()
        def math_func(expr, value):
            x = sp.Symbol('x')
            result = expr.subs(x, value)
            return {
                "input_expr": expr,
                "symbol": x,
                "value": value,
                "result": result,
                "symbols": {x: value}
            }

        # Create test data
        x = sp.Symbol('x')
        expr = x**2 + 2*x + 1

        # Call the function
        result = math_func(expr, 2)

        # Verify the function result - should be 9 (2^2 + 2*2 + 1)
        assert result["result"] == 9

        # Get the debug file and verify serialization
        _, actual_data = get_debug_file(get_test_hash())

        # Verify the structure
        assert "FUNCTION" in actual_data
        assert "INPUTS" in actual_data
        assert "OUTPUT" in actual_data

        # Verify the output contains serialized SymPy objects
        output = actual_data["OUTPUT"]
        assert isinstance(output["input_expr"], str)
        assert isinstance(output["symbol"], str)
        assert output["value"] == 2

        # The result is serialized as a string, so we need to check the string value
        assert output["result"] == "9" or output["result"] == 9

    @patch("detective.snapshot._generate_short_hash")
    def test_complex_nested_structures(self, mock_hash):
        """Test handling of complex nested structures with SymPy objects."""
        mock_hash.return_value = get_test_hash()

        x, y, z = sp.symbols('x y z')

        # Create a complex nested structure
        test_data = {
            "symbols": [x, y, z],
            "expressions": {
                x: x**2 + y,
                y: y**2 + z,
                z: z**2 + x
            },
            "nested": {
                "level1": {
                    x: {
                        y: {
                            z: x + y + z
                        }
                    }
                }
            },
            "mixed": [
                {x: 1},
                {y: [x, y, z]},
                {z: {"expr": x*y*z}}
            ]
        }

        @snapshot()
        def process_complex_data(data):
            return data

        result = process_complex_data(test_data)

        # Verify the function returned the original data
        assert result == test_data

        # Get the debug file and verify serialization
        _, actual_data = get_debug_file(get_test_hash())

        # Convert the actual data to JSON and back to verify serialization
        json_str = json.dumps(actual_data, cls=CustomJSONEncoder)
        decoded = json.loads(json_str)

        # Verify the structure
        inputs_data = decoded["INPUTS"]["data"]
        assert isinstance(inputs_data["symbols"], list)
        assert "x" in inputs_data["expressions"]
        assert isinstance(inputs_data["nested"]["level1"]["x"]["y"]["z"], str)
        assert isinstance(inputs_data["mixed"][1]["y"], list)

    @patch("detective.snapshot._generate_short_hash")
    def test_error_handling(self, mock_hash):
        """Test error handling in JSON serialization."""
        mock_hash.return_value = get_test_hash()

        class UnserializableObject:
            def __repr__(self):
                return "UnserializableObject()"

        # Create test data with problematic objects
        test_data = {
            "normal": "value",
            "problematic": UnserializableObject(),
            "nested": {
                "more_problems": [UnserializableObject(), UnserializableObject()],
                "normal_list": [1, 2, 3]
            }
        }

        @snapshot()
        def process_problematic_data(data):
            return data

        process_problematic_data(test_data)

        # Get the debug file and verify serialization
        _, actual_data = get_debug_file(get_test_hash())

        # Verify the structure
        assert "FUNCTION" in actual_data
        assert "INPUTS" in actual_data
        assert "OUTPUT" in actual_data

        # The problematic objects should be serialized in some way (either as strings or empty objects)
        # Let's check that the normal values are preserved correctly
        inputs = actual_data["INPUTS"]["data"]
        assert inputs["normal"] == "value"
        assert inputs["nested"]["normal_list"] == [1, 2, 3]

        # Verify that the problematic objects are handled in some way
        assert "problematic" in inputs
        assert "more_problems" in inputs["nested"]

    @patch("detective.snapshot._generate_short_hash")
    def test_snapshot_file_content(self, mock_hash):
        """Test that snapshot files contain the expected content."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def test_func(x, y):
            return {"result": x + y, "symbol": sp.Symbol('x')}

        result = test_func(1, 2)

        # Verify the function result
        assert result["result"] == 3
        assert isinstance(result["symbol"], sp.Symbol)

        # Get the debug file and verify content
        _, actual_data = get_debug_file(get_test_hash())

        # Verify structure and content
        assert actual_data["FUNCTION"] == "test_func"
        assert "INPUTS" in actual_data
        assert "OUTPUT" in actual_data
        assert actual_data["INPUTS"]["x"] == 1
        assert actual_data["INPUTS"]["y"] == 2
        assert actual_data["OUTPUT"]["result"] == 3
        assert isinstance(actual_data["OUTPUT"]["symbol"], str)
