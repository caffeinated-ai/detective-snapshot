import os
import uuid
from unittest.mock import patch

import pytest

from detective import snapshot
from detective.snapshot import inner_calls_var, session_id_var

from .fixtures_data import CocoCat
from .utils import (
    are_snapshots_equal,
    get_debug_file,
    get_test_hash,
    get_test_name,
    load_expected_data,
    mock_hash_sequence,
    setup_debug_dir,
    update_expected_data,
)


# Innermost function
@snapshot()
def do(activity):
    return activity["name"]


# Inner function
@snapshot()
def play(activities):
    for activity in activities:
        do(activity)  # Innermost function call
    return len(activities)


@snapshot()
def eat(foods):
    return foods[0]


# Outermost function
@snapshot()
def life(cat):
    cat_name = cat["name"]
    eat(cat["foods"])
    play(cat["activities"])  # Inner function call
    return (cat_name, "is", "happy")


class TestSnapshotFunctionNesting:
    def setup_method(self):
        """Setup before each test method."""
        setup_debug_dir()

    @patch("detective.snapshot._generate_short_hash")
    def test_multiple_nested_levels(self, mock_hash, update_snapshots):
        """Test multiple levels of nested function calls."""
        mock_hash.return_value = get_test_hash()

        result = life(CocoCat)
        assert result == ("Coco", "is", "happy")

        filepath, actual_data = get_debug_file(get_test_hash())

        test_name = get_test_name()
        if update_snapshots:
            update_expected_data(filepath, test_name)
            return

        # Load expected data
        expected_data = load_expected_data(test_name)
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_no_inner_calls(self, mock_hash):
        """Test function with no inner calls."""
        mock_hash.return_value = get_test_hash()

        foods = ["sushi", "salmon", "tuna"]
        result = eat(foods)
        assert result == "sushi"

        _, actual_data = get_debug_file(get_test_hash())

        expected_data = {
            "FUNCTION": "eat",
            "INPUTS": {"foods": ["sushi", "salmon", "tuna"]},
            "OUTPUT": "sushi",
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_multiple_nested_levels_simple(self, mock_hash):
        """Test simple nested function calls with numerical operations."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def level1(x):
            return level2(x + 1) * 2

        @snapshot()
        def level2(y):
            return level3(y + 2) + 3

        @snapshot()
        def level3(z):
            return z * 4

        result = level1(1)  # 1 -> 2 -> 4 -> 16 + 3 -> 19 * 2 -> 38
        assert result == 38

        _, actual_data = get_debug_file(get_test_hash())

        expected_data = {
            "FUNCTION": "level1",
            "INPUTS": {"x": 1},
            "OUTPUT": 38,
            "CALLS": [
                {
                    "FUNCTION": "level2",
                    "INPUTS": {"y": 2},
                    "OUTPUT": 19,
                    "CALLS": [{"FUNCTION": "level3", "INPUTS": {"z": 4}, "OUTPUT": 16}],
                }
            ],
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_mixed_method_types_nested(self, mock_hash):
        """Test nested calls with mixed method types (class, instance, static)."""
        mock_hash.return_value = get_test_hash()

        class Calculator:
            base = 10

            def __init__(self, multiplier):
                self.multiplier = multiplier

            @classmethod
            @snapshot()
            def outer_class_method(cls, x):
                calc = Calculator(2)
                return calc.middle_instance_method(x + cls.base)

            @snapshot()
            def middle_instance_method(self, y):
                result = self.inner_static_method(y * self.multiplier)
                return result + 5

            @staticmethod
            @snapshot()
            def inner_static_method(z):
                return z * 3

        result = Calculator.outer_class_method(1)
        assert result == 71

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "outer_class_method",
            "INPUTS": {"cls": {"base": 10}, "x": 1},
            "OUTPUT": 71,
            "CALLS": [
                {
                    "FUNCTION": "middle_instance_method",
                    "INPUTS": {"self": {"multiplier": 2}, "y": 11},
                    "OUTPUT": 71,
                    "CALLS": [
                        {
                            "FUNCTION": "inner_static_method",
                            "INPUTS": {"z": 22},
                            "OUTPUT": 66,
                        }
                    ],
                }
            ],
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_multiple_calls_nested_function(self, mock_hash):
        """Test nested function being called multiple times with different parameters."""
        mock_hash.side_effect = mock_hash_sequence(2)  # Get two hashes

        @snapshot()
        def outer(x):
            first_result = inner(x)
            second_result = inner(x * 2)
            return first_result + second_result

        @snapshot()
        def inner(y):
            return y * 10

        # First call with x=5
        result1 = outer(
            5
        )  # First call: 5 * 10 = 50, Second call: 10 * 10 = 100, Total: 150
        assert result1 == 150

        # Second call with x=3
        result2 = outer(
            3
        )  # First call: 3 * 10 = 30, Second call: 6 * 10 = 60, Total: 90
        assert result2 == 90

        # Check first call's debug file
        _, actual_data1 = get_debug_file(get_test_hash())
        expected_data1 = {
            "FUNCTION": "outer",
            "INPUTS": {"x": 5},
            "OUTPUT": 150,
            "CALLS": [
                {"FUNCTION": "inner", "INPUTS": {"y": 5}, "OUTPUT": 50},
                {"FUNCTION": "inner", "INPUTS": {"y": 10}, "OUTPUT": 100},
            ],
        }
        assert are_snapshots_equal(actual_data1, expected_data1)

        # Check second call's debug file
        _, actual_data2 = get_debug_file(get_test_hash("second"))
        expected_data2 = {
            "FUNCTION": "outer",
            "INPUTS": {"x": 3},
            "OUTPUT": 90,
            "CALLS": [
                {"FUNCTION": "inner", "INPUTS": {"y": 3}, "OUTPUT": 30},
                {"FUNCTION": "inner", "INPUTS": {"y": 6}, "OUTPUT": 60},
            ],
        }
        assert are_snapshots_equal(actual_data2, expected_data2)

        # Verify that two debug files were created
        debug_dir = os.path.join(os.getcwd(), "_snapshots")
        debug_files = sorted(
            [f for f in os.listdir(debug_dir) if f.startswith("outer_")]
        )
        assert len(debug_files) == 2

        # Verify the filenames match our expected pattern
        expected_hashes = [get_test_hash(), get_test_hash("second")]
        assert all(
            any(f.endswith(f"_{h}.json") for h in expected_hashes) for f in debug_files
        ), f"Files {debug_files} don't match expected hash patterns"

    @patch("detective.snapshot._generate_short_hash")
    def test_partial_nested_decoration(self, mock_hash):
        """Test when only outer and innermost functions are decorated with @snapshot."""
        mock_hash.return_value = get_test_hash()

        @snapshot()
        def outer_decorated(x):
            # Call undecorated middle function
            return middle_undecorated(x + 1)

        def middle_undecorated(y):
            # Call decorated innermost function
            return inner_decorated(y * 2)

        @snapshot()
        def inner_decorated(z):
            return z + 5

        result = outer_decorated(10)  # 10 -> 11 -> 22 -> 27
        assert result == 27

        # Check debug file
        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "outer_decorated",
            "INPUTS": {"x": 10},
            "OUTPUT": 27,
            "CALLS": [
                {
                    "FUNCTION": "inner_decorated",
                    "INPUTS": {"z": 22},
                    "OUTPUT": 27,
                }
            ],
        }
        assert are_snapshots_equal(actual_data, expected_data)
