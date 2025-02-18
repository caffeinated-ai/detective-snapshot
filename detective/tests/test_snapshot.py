import unittest
from dataclasses import dataclass
from typing import List

from detective.snapshot import _get_nested_value


@dataclass
class TestItem:
    normalized_text: str
    full_text: str


@dataclass
class TestContext:
    names: List[str]


class TestDebugUtils(unittest.TestCase):
    def test_direct_parameter_access(self):
        """Test direct parameter access without $ prefix"""
        test_obj = {
            "kwargs": {
                "param1": "value1",
                "param2": "value2",
            }
        }
        self.assertEqual(_get_nested_value(test_obj, "param1"), "value1")

    def test_nested_parameter_access(self):
        """Test nested parameter access with dot notation"""
        test_obj = {
            "kwargs": {
                "context": TestContext(names=["name1", "name2"]),
            }
        }
        self.assertEqual(
            _get_nested_value(test_obj, "context.names"), ["name1", "name2"]
        )

    def test_array_access(self):
        """Test array indexing"""
        test_obj = {
            "kwargs": {
                "items": [
                    TestItem("text1", "full1"),
                    TestItem("text2", "full2"),
                ]
            }
        }
        self.assertEqual(
            _get_nested_value(test_obj, "items[0].normalized_text"), "text1"
        )

    def test_array_wildcard_access(self):
        """Test array wildcard access"""
        test_obj = {
            "kwargs": {
                "result_items": [
                    TestItem("text1", "full1"),
                    TestItem("text2", "full2"),
                ]
            }
        }
        expected = ["text1", "text2"]
        result = _get_nested_value(test_obj, "result_items[*].normalized_text")
        self.assertEqual(result, expected)

    def test_array_wildcard_with_empty_list(self):
        """Test array wildcard with empty list"""
        test_obj = {"kwargs": {"result_items": []}}
        result = _get_nested_value(test_obj, "result_items[*].normalized_text")
        self.assertEqual(result, [])

    def test_array_wildcard_with_none_values(self):
        """Test array wildcard with None values"""
        test_obj = {
            "kwargs": {
                "result_items": [
                    TestItem("text1", "full1"),
                    None,
                    TestItem("text3", "full3"),
                ]
            }
        }
        expected = ["text1", None, "text3"]
        result = _get_nested_value(test_obj, "result_items[*].normalized_text")
        self.assertEqual(result, expected)

    def test_complex_nested_array_access(self):
        """Test complex nested array access with multiple wildcards"""
        test_obj = {
            "kwargs": {
                "groups": [
                    {
                        "items": [
                            TestItem("text1", "full1"),
                            TestItem("text2", "full2"),
                        ]
                    },
                    {
                        "items": [
                            TestItem("text3", "full3"),
                        ]
                    },
                ]
            }
        }
        expected = [["text1", "text2"], ["text3"]]
        result = _get_nested_value(test_obj, "groups[*].items[*].normalized_text")
        self.assertEqual(result, expected)

    def test_multiple_field_selection(self):
        """Test selecting multiple fields with parentheses syntax"""
        test_obj = {
            "kwargs": {
                "items": [
                    TestItem("text1", "full1"),
                    TestItem("text2", "full2"),
                ]
            }
        }
        result = _get_nested_value(test_obj, "items[*].(normalized_text,full_text)")
        expected = [
            {"normalized_text": "text1", "full_text": "full1"},
            {"normalized_text": "text2", "full_text": "full2"},
        ]
        self.assertEqual(result, expected)

    def test_multiple_field_selection_with_spaces(self):
        """Test selecting multiple fields with spaces in parentheses syntax"""
        test_obj = {
            "kwargs": {
                "result_items": [
                    TestItem("text1", "full1"),
                    TestItem("text2", "full2"),
                ]
            }
        }
        # Test with spaces after comma
        result1 = _get_nested_value(
            test_obj, "result_items[*].(normalized_text, full_text)"
        )
        expected = [
            {"normalized_text": "text1", "full_text": "full1"},
            {"normalized_text": "text2", "full_text": "full2"},
        ]
        self.assertEqual(result1, expected)

        # Test with spaces and no spaces
        result2 = _get_nested_value(
            test_obj, "result_items[*].(normalized_text,full_text, other_field)"
        )
        expected = [
            {"normalized_text": "text1", "full_text": "full1", "other_field": None},
            {"normalized_text": "text2", "full_text": "full2", "other_field": None},
        ]
        self.assertEqual(result2, expected)

        # Test with mixed spacing
        result3 = _get_nested_value(
            test_obj, "result_items[*].( normalized_text,full_text )"
        )
        expected = [
            {"normalized_text": "text1", "full_text": "full1"},
            {"normalized_text": "text2", "full_text": "full2"},
        ]
        self.assertEqual(result3, expected)

    def test_real_world_example(self):
        """Test with real-world example from SearchResultAnalyzer"""
        test_obj = {
            "kwargs": {
                "result_items": [
                    TestItem(normalized_text="Clean Product 1", full_text="Product 1"),
                    TestItem(normalized_text="Clean Product 2", full_text="Product 2"),
                ],
                "context": TestContext(names=["Previous Product"]),
            }
        }

        # Test result_items array access
        result1 = _get_nested_value(test_obj, "result_items[*].normalized_text")
        self.assertEqual(result1, ["Clean Product 1", "Clean Product 2"])

        # Test context access
        result2 = _get_nested_value(test_obj, "context.names")
        self.assertEqual(result2, ["Previous Product"])

    def test_invalid_path(self):
        """Test handling of invalid paths"""
        test_obj = {"kwargs": {"param1": "value1"}}
        self.assertIsNone(_get_nested_value(test_obj, "invalid.path"))
        self.assertIsNone(_get_nested_value(test_obj, "param1[999]"))
        self.assertIsNone(_get_nested_value(test_obj, "param1.nonexistent"))

    def test_none_handling(self):
        """Test handling of None values at different levels"""
        test_obj = {
            "kwargs": {
                "items": [
                    {"value": None},
                    None,
                    {"value": "test"},
                ]
            }
        }
        result = _get_nested_value(test_obj, "items[*].value")
        self.assertEqual(result, [None, None, "test"])

    def test_multiple_field_selection_real_world(self):
        """Test the exact pattern from product_search_service.py"""
        test_obj = {
            "kwargs": {
                "result_items": [
                    TestItem(normalized_text="Product 1", full_text="Original 1"),
                    TestItem(normalized_text="Product 2", full_text="Original 2"),
                ]
            }
        }

        # Test the exact pattern from product_search_service.py
        result = _get_nested_value(
            test_obj, "result_items[*].(normalized_text, full_text)"
        )
        expected = [
            {"normalized_text": "Product 1", "full_text": "Original 1"},
            {"normalized_text": "Product 2", "full_text": "Original 2"},
        ]
        self.assertEqual(result, expected)

        # Also test individual field access to compare
        result_normalized = _get_nested_value(
            test_obj, "result_items[*].normalized_text"
        )
        result_full = _get_nested_value(test_obj, "result_items[*].full_text")
        self.assertEqual(result_normalized, ["Product 1", "Product 2"])
        self.assertEqual(result_full, ["Original 1", "Original 2"])


if __name__ == "__main__":
    unittest.main()
