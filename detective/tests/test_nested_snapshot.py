import json
import os
import unittest
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import List
from unittest.mock import MagicMock, patch

from detective import snapshot
from detective.snapshot import inner_calls_var, session_id_var


# Mock protobuf class for testing
class MockProto:
    def __init__(self, value):
        self.value = value
        self.DESCRIPTOR = MagicMock()  # Mock the DESCRIPTOR attribute


@dataclass
class SearchContext:
    """Test dataclass for serialization testing"""

    query: str
    filters: list
    page: int = 1


class ResultType(Enum):
    """Test enum for serialization testing"""

    PRODUCT = "product"
    CATEGORY = "category"


@dataclass
class TestItem:
    normalized_text: str
    full_text: str


@dataclass
class TestContext:
    names: List[str]


class TestDebugUtilsNested(unittest.TestCase):
    def setUp(self):
        # Ensure debug mode is on for these tests
        os.environ["DEBUG"] = "true"
        # Clean up any existing debug output
        self.debug_dir = os.path.join(os.getcwd(), "debug_output")
        if os.path.exists(self.debug_dir):
            for file in os.listdir(self.debug_dir):
                os.remove(os.path.join(self.debug_dir, file))
        else:
            os.makedirs(self.debug_dir)

    def tearDown(self):
        # Clean up after tests
        if os.path.exists(self.debug_dir):
            for file in os.listdir(self.debug_dir):
                os.remove(os.path.join(self.debug_dir, file))
            os.rmdir(self.debug_dir)
        os.environ["DEBUG"] = "false"  # Reset debug mode
        # Reset context variables
        session_id_var.set(None)
        inner_calls_var.set(None)

    @patch("detective.snapshot.uuid.uuid4")
    def test_nested_function_calls(self, mock_uuid):
        # Mock uuid4 to return a consistent value for predictable filenames
        mock_uuid_str = "12345678123456781234567812345678"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot(
            input_fields=["kwargs.input_a", "kwargs.input_b"], output_fields=["result"]
        )
        def outer_function(input_a, input_b):
            inner_result = inner_function(input_a + 1)
            return input_b + inner_result

        @snapshot(input_fields=["kwargs.input_c"], output_fields=["interim_result"])
        def inner_function(input_c):
            return input_c * 2

        # Call the outer function
        result = outer_function(input_a=5, input_b=10)
        self.assertEqual(result, 22)  # Check the actual result

        # Find the debug output file
        debug_files = os.listdir(self.debug_dir)
        self.assertEqual(len(debug_files), 1)  # Expect one file
        filepath = os.path.join(self.debug_dir, debug_files[0])

        # Load and check the JSON data
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Assertions about the structure
        self.assertIn("input", data)
        self.assertIn("output", data)
        self.assertIn("inner_function_calls", data)
        self.assertEqual(len(data["inner_function_calls"]), 1)

        # Check outer function input/output
        self.assertEqual(data["input"], {"kwargs": {"input_a": 5, "input_b": 10}})
        self.assertEqual(data["output"], {"result": 22})

        # Check inner function input/output
        inner_call = data["inner_function_calls"][0]
        self.assertEqual(inner_call["function_name"], "inner_function")
        self.assertEqual(inner_call["input"], {"kwargs": {"input_c": 6}})
        self.assertEqual(inner_call["output"], {"interim_result": 12})

    @patch("detective.snapshot.uuid.uuid4")
    def test_no_inner_calls(self, mock_uuid):
        mock_uuid_str = "23456789234567892345678923456789"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot(input_fields=["kwargs.x"], output_fields=["result"])
        def simple_function(x):
            return x * 2

        result = simple_function(x=10)
        self.assertEqual(result, 20)

        debug_files = os.listdir(self.debug_dir)
        self.assertEqual(len(debug_files), 1)
        filepath = os.path.join(self.debug_dir, debug_files[0])

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("input", data)
        self.assertIn("output", data)
        self.assertIn("inner_function_calls", data)
        self.assertEqual(len(data["inner_function_calls"]), 0)  # No inner calls
        self.assertEqual(data["input"], {"kwargs": {"x": 10}})
        self.assertEqual(data["output"], {"result": 20})

    @patch("detective.snapshot.uuid.uuid4")
    def test_exception_handling(self, mock_uuid):
        mock_uuid_str = "34567890345678903456789034567890"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def function_with_exception(x):
            raise ValueError("Something went wrong")

        with self.assertRaises(ValueError):
            function_with_exception(x=5)

        debug_files = os.listdir(self.debug_dir)
        self.assertEqual(len(debug_files), 1)
        filepath = os.path.join(self.debug_dir, debug_files[0])

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("input", data)
        self.assertIn("output", data)
        self.assertIn("inner_function_calls", data)
        self.assertEqual(len(data["inner_function_calls"]), 0)
        self.assertIn("kwargs", data["input"])
        self.assertEqual(data["input"]["kwargs"]["x"], 5)
        self.assertIn("error", data["output"])
        self.assertEqual(data["output"]["error"], "Something went wrong")

    @patch("detective.snapshot.uuid.uuid4")
    def test_multiple_nested_levels(self, mock_uuid):
        mock_uuid_str = "45678901456789014567890145678901"
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
        self.assertEqual(result, 38)

        debug_files = os.listdir(self.debug_dir)
        filepath = os.path.join(self.debug_dir, debug_files[0])

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(len(data["inner_function_calls"]), 2)
        self.assertEqual(data["inner_function_calls"][0]["function_name"], "level2")
        self.assertEqual(data["inner_function_calls"][1]["function_name"], "level3")
        # Level 2 input
        self.assertEqual(data["inner_function_calls"][0]["input"]["kwargs"]["y"], 2)
        # Level 3 input
        self.assertEqual(data["inner_function_calls"][1]["input"]["kwargs"]["z"], 4)
        # Level 3 output
        self.assertEqual(data["inner_function_calls"][1]["output"], 16)
        # Level 2 output
        self.assertEqual(data["inner_function_calls"][0]["output"], 19)
        # Level 1 output
        self.assertEqual(data["output"], 38)

    def test_debug_mode_off(self):
        os.environ["DEBUG"] = "false"

        @snapshot()
        def some_function(x):
            return x * 2

        result = some_function(5)
        self.assertEqual(result, 10)
        self.assertEqual(
            len(os.listdir(self.debug_dir)), 0
        )  # No files should be created

    @patch("detective.snapshot.uuid.uuid4")
    def test_dataclass_serialization(self, mock_uuid):
        """Test serialization of dataclasses and enums"""
        mock_uuid_str = "56789012567890125678901256789012"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def search_function(context: SearchContext, result_type: ResultType):
            return {"matches": ["result1", "result2"], "type": result_type}

        # Create test data
        context = SearchContext(
            query="test query", filters=["filter1", "filter2"], page=1
        )

        result = search_function(context, ResultType.PRODUCT)
        self.assertEqual(result["type"], ResultType.PRODUCT)

        # Check debug output
        debug_files = os.listdir(self.debug_dir)
        self.assertEqual(len(debug_files), 1)
        filepath = os.path.join(self.debug_dir, debug_files[0])

        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        # Verify dataclass serialization
        self.assertEqual(
            data["input"]["kwargs"]["context"],
            {"query": "test query", "filters": ["filter1", "filter2"], "page": 1},
        )
        # Verify enum serialization
        self.assertEqual(data["output"]["type"], "product")

    @patch("detective.snapshot.uuid.uuid4")
    def test_protobuf_serialization(self, mock_uuid):
        """Test serialization of protobuf objects"""
        mock_uuid_str = "67890123678901236789012367890123"
        mock_uuid.return_value = uuid.UUID(mock_uuid_str)

        @snapshot()
        def proto_function(proto_obj):
            return {"result": proto_obj}

        # Create a mock protobuf object
        proto = MockProto({"field1": "value1", "field2": 42})

        # Mock protobuf_to_json to return a known JSON string
        with patch("detective._to_json_compatible") as mock_to_json:
            mock_to_json.return_value = {"field1": "value1", "field2": 42}

            result = proto_function(proto)
            self.assertEqual(result["result"], proto)

            # Check debug output
            debug_files = os.listdir(self.debug_dir)
            self.assertEqual(len(debug_files), 1)
            filepath = os.path.join(self.debug_dir, debug_files[0])

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            # Verify protobuf serialization
            self.assertEqual(
                data["output"]["result"], {"field1": "value1", "field2": 42}
            )
