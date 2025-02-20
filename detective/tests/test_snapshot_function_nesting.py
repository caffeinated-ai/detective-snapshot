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
    get_test_name,
    load_expected_data,
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


@pytest.fixture(autouse=True)
def setup_module():
    """Setup for each test."""
    os.environ["DEBUG"] = "true"
    # Clean up any existing debug output
    debug_dir = os.path.join(os.getcwd(), "debug_snapshots")
    if os.path.exists(debug_dir):
        for file in os.listdir(debug_dir):
            os.remove(os.path.join(debug_dir, file))
    else:
        os.makedirs(debug_dir)
    yield
    session_id_var.set(None)
    inner_calls_var.set([])


class TestSnapshotFunctionNesting:  # Renamed class
    def setup_method(self):
        """Setup before each test method."""
        setup_debug_dir()

    @patch("detective.snapshot.uuid.uuid4")
    def test_multiple_nested_levels(self, mock_uuid, update_snapshots):
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        result = life(CocoCat)
        assert result == ("Coco", "is", "happy")

        filepath, actual_data = get_debug_file(mock_uuid_str)

        test_name = get_test_name()
        if update_snapshots:
            update_expected_data(filepath, test_name)
            return

        # Load expected data
        expected_data = load_expected_data(test_name)
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_no_inner_calls(self, mock_uuid):
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        foods = ["sushi", "salmon", "tuna"]
        result = eat(foods)
        assert result == "sushi"

        _, actual_data = get_debug_file(mock_uuid_str)

        expected_data = {
            "FUNCTION": "eat",
            "INPUTS": {"foods": ["sushi", "salmon", "tuna"]},
            "OUTPUT": "sushi",
        }

        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot.uuid.uuid4")
    def test_multiple_nested_levels_simple(self, mock_uuid):
        """Test simple nested function calls with numerical operations."""
        mock_uuid_str = "45678901-4567-8901-4567-890145678901"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

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

        _, actual_data = get_debug_file(mock_uuid_str)

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

    @patch("detective.snapshot.uuid.uuid4")
    def test_exception_handling(self, mock_uuid, update_snapshots):
        """Test that exceptions in inner functions are properly captured."""
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

        # Call the function and expect an exception
        with pytest.raises(ValueError) as exc_info:
            outer()
        assert str(exc_info.value) == "Something went wrong!"

        # Check the debug snapshot
        filepath, actual_data = get_debug_file(mock_uuid_str)

        test_name = get_test_name()
        if update_snapshots:
            update_expected_data(filepath, test_name)
            return

        expected_data = load_expected_data(test_name)
        assert are_snapshots_equal(actual_data, expected_data)
