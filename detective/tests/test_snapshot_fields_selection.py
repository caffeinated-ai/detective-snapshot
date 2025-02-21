import os
import uuid
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import patch

import pytest

from detective import snapshot

from .fixtures_data import (
    BoboCat,
    BoboProto,
    CatData_dict,
    CocoCat,
    CocoDataclass,
    CocoProto,
    JaggerCat,
    JaggerDataclass,
    JaggerProto,
)
from .utils import are_snapshots_equal, get_debug_file, setup_debug_dir, get_test_hash

# Split the test cases into arrays based on input patterns
SINGLE_INPUT_TEST_CASES = [
    {
        "name": "wildcard_single_field",
        "input_fields": ["cats.*.color"],
        "expected_input": {
            "cats": {
                "Coco": {"color": "calico"},
                "Bobo": {"color": "tuxedo"},
                "Jagger": {"color": "void"},
            }
        },
    },
    {
        "name": "wildcard_single_field_args_syntax",
        "input_fields": ["args[0].*.color"],
        "expected_input": {
            "cats": {  # Note: still using "cats" as the key, not "args[0]"
                "Coco": {"color": "calico"},
                "Bobo": {"color": "tuxedo"},
                "Jagger": {"color": "void"},
            }
        },
    },
    {
        "name": "nested_array_fields",
        "input_fields": [
            "cats.*.activities[*].name",
            "cats.*.activities[*].adorableness",
        ],
        "expected_input": {
            "cats": {
                "Coco": {
                    "activities": [
                        {"name": "sunbathing"},
                        {"name": "brushing", "adorableness": "melts_like_butter"},
                    ]
                },
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "specific_array_index",
        "input_fields": ["cats.*.activities[0].name"],
        "expected_input": {
            "cats": {
                "Coco": {"activities": [{"name": "sunbathing"}]},
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {"activities": [{"name": "shadow prowling"}]},
            }
        },
    },
    {
        "name": "multiple_field_selection",
        "input_fields": ["cats.Coco.(color, name)"],
        "expected_input": {"cats": {"Coco": {"name": "Coco", "color": "calico"}}},
    },
    {
        "name": "multiple_cats_specific_fields",
        "input_fields": ["cats.(Coco, Bobo).color"],
        "expected_input": {
            "cats": {"Coco": {"color": "calico"}, "Bobo": {"color": "tuxedo"}}
        },
    },
    {
        "name": "nested_multiple_selection",
        "input_fields": ["cats.*.activities[*].(name, cuteness)"],
        "expected_input": {
            "cats": {
                "Coco": {
                    "activities": [
                        {"name": "sunbathing", "cuteness": "purrfectly_toasty"},
                        {"name": "brushing"},
                    ]
                },
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "overlapping_fields_specific_and_wildcard",
        "input_fields": [
            "cats.Coco.activities[*].name",  # Specific cat
            "cats.*.activities[*].name",  # All cats
        ],
        "expected_input": {
            "cats": {
                "Coco": {
                    "activities": [
                        {"name": "sunbathing"},
                        {"name": "brushing"},
                    ]
                },
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "duplicate_fields",
        "input_fields": ["cats.*.color", "cats.*.color"],  # Same field twice
        "expected_input": {
            "cats": {
                "Coco": {"color": "calico"},
                "Bobo": {"color": "tuxedo"},
                "Jagger": {"color": "void"},
            }
        },
    },
    {
        "name": "overlapping_nested_fields",
        "input_fields": [
            "cats.*.activities[0]",  # First activity of each cat
            "cats.*.activities[*].name",  # All activity names
        ],
        "expected_input": {
            "cats": {
                "Coco": {
                    "activities": [
                        {"name": "sunbathing", "cuteness": "purrfectly_toasty"},
                        {"name": "brushing"},
                    ]
                },
                "Bobo": {
                    "activities": [
                        {"name": "belly rubs", "goofiness": "rolls_around_happily"}
                    ]
                },
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling", "stealth": "ninja_level"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "overlapping_array_indices",
        "input_fields": [
            "cats.*.activities[0].name",  # First activity name
            "cats.*.activities[*].name",  # All activity names
        ],
        "expected_input": {
            "cats": {
                "Coco": {"activities": [{"name": "sunbathing"}, {"name": "brushing"}]},
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "overlapping_args_and_direct_syntax",
        "input_fields": [
            "args[0].Coco.activities[*].name",  # Using args[0] syntax
            "cats.*.activities[*].name",  # Using parameter name directly
        ],
        "expected_input": {
            "cats": {
                "Coco": {
                    "activities": [
                        {"name": "sunbathing"},
                        {"name": "brushing"},
                    ]
                },
                "Bobo": {"activities": [{"name": "belly rubs"}]},
                "Jagger": {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            }
        },
    },
    {
        "name": "args_nested_wildcard_fields",
        "input_fields": [
            "args[0].*.color",
            "cats.*.name",
        ],
        "expected_input": {
            "cats": {  # Note: still using "cats" as the key
                "Coco": {"color": "calico", "name": "Coco"},
                "Bobo": {"color": "tuxedo", "name": "Bobo"},
                "Jagger": {"color": "void", "name": "Jagger"},
            }
        },
    },
]

ARGS_TEST_CASES = [
    {
        "name": "multiple_args_selection",
        "input_fields": ["args[0].name", "args[1].color"],
        "args": [CocoCat, BoboCat],
        "expected_input": {"args": [{"name": "Coco"}, {"color": "tuxedo"}]},
    },
    {
        "name": "args_and_kwargs_mixed",
        "input_fields": ["args[0].name", "other_cat.color"],
        "args": [CocoCat],
        "kwargs": {"other_cat": BoboCat},
        "expected_input": {
            "args": [{"name": "Coco"}],
            "kwargs": {"other_cat": {"color": "tuxedo"}},
        },
    },
]

# Add after ARGS_TEST_CASES
ARRAY_TEST_CASES = [
    {
        "name": "array_of_dicts",
        "input_fields": ["cats[*].color", "cats[*].name"],
        "input_data": [
            {"name": "Coco", "color": "calico"},
            {"name": "Bobo", "color": "tuxedo"},
            {"name": "Jagger", "color": "void"},
        ],
        "expected_input": {
            "cats": [
                {"name": "Coco", "color": "calico"},
                {"name": "Bobo", "color": "tuxedo"},
                {"name": "Jagger", "color": "void"},
            ]
        },
    },
    {
        "name": "array_of_objects",
        "input_fields": ["cats[*].(name, color)"],
        "input_data": [CocoCat, BoboCat, JaggerCat],
        "expected_input": {
            "cats": [
                {"name": "Coco", "color": "calico"},
                {"name": "Bobo", "color": "tuxedo"},
                {"name": "Jagger", "color": "void"},
            ]
        },
    },
    {
        "name": "array_nested_fields",
        "input_fields": ["cats[*].activities[*].name"],
        "input_data": [CocoCat, BoboCat, JaggerCat],
        "expected_input": {
            "cats": [
                {"activities": [{"name": "sunbathing"}, {"name": "brushing"}]},
                {"activities": [{"name": "belly rubs"}]},
                {
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ]
                },
            ]
        },
    },
    {
        "name": "array_specific_indices",
        "input_fields": ["cats[0].activities[1].name", "cats[2].activities[0].name"],
        "input_data": [CocoCat, BoboCat, JaggerCat],
        "expected_input": {
            "cats": [
                {"activities": [{"name": "brushing"}]},
                # Middle element omitted entirely since it's not selected
                {"activities": [{"name": "shadow prowling"}]},
            ]
        },
    },
]

# If you have protobuf test cases, add them here:
PROTO_ARRAY_TEST_CASES = [
    # Add when proto fixtures are available
]


# Add these dataclass definitions at the top
@dataclass
class Activity:
    name: str
    fun_level: int


MIXED_OBJECTS_TEST_CASES = [
    {
        "name": "dict_of_protos",
        "input_fields": ["data.cats.*.name", "data.cats.*.color"],
        "input_data": {
            "cats": {
                "cat1": CocoProto,
                "cat2": BoboProto,
            }
        },
        "expected_input": {
            "data": {
                "cats": {
                    "cat1": {"name": "Coco", "color": "calico"},
                    "cat2": {"name": "Bobo", "color": "tuxedo"},
                }
            }
        },
    },
    {
        "name": "array_of_protos",
        "input_fields": ["data.cats[*].(name, color)"],
        "input_data": {
            "cats": [
                CocoProto,
                BoboProto,
                JaggerProto,
            ]
        },
        "expected_input": {
            "data": {
                "cats": [
                    {"name": "Coco", "color": "calico"},
                    {"name": "Bobo", "color": "tuxedo"},
                    {"name": "Jagger", "color": "void"},
                ]
            }
        },
    },
    {
        "name": "mixed_dataclass_with_proto",
        "input_fields": [
            "data.*.name",
            "data.*.color",
            "data.*.activities[*].name",
        ],
        "input_data": {"cat1": CocoCat, "cat2": BoboProto, "cat3": JaggerDataclass},
        "expected_input": {
            "data": {
                "cat1": {
                    "name": "Coco",
                    "color": "calico",
                    "activities": [{"name": "sunbathing"}, {"name": "brushing"}],
                },
                "cat2": {
                    "name": "Bobo",
                    "color": "tuxedo",
                    "activities": [{"name": "belly rubs"}],
                },
                "cat3": {
                    "name": "Jagger",
                    "color": "void",
                    "activities": [
                        {"name": "shadow prowling"},
                        {"name": "shoulder rides"},
                    ],
                },
            }
        },
    },
    {
        "name": "array_of_mixed_objects",
        "input_fields": [
            "data.cats[*].(name, color)",
            "data.cats[*].activities",
        ],
        "input_data": {
            "cats": [
                {
                    "name": "Coco",
                    "color": "calico",
                    "activities": CocoProto.activities,
                },
                BoboProto,
            ]
        },
        "expected_input": {
            "data": {
                "cats": [
                    {
                        "name": "Coco",
                        "color": "calico",
                        "activities": (
                            '[name: "sunbathing"\ncuteness: "purrfectly_toasty"\n, '
                            'name: "brushing"\nadorableness: "melts_like_butter"\n]'
                        ),
                    },
                    {
                        "name": "Bobo",
                        "color": "tuxedo",
                        "activities": [
                            {"name": "belly rubs", "goofiness": "rolls_around_happily"}
                        ],
                    },
                ]
            }
        },
    },
    {
        "name": "explicit_nesting_levels",
        "input_fields": ["data.cat.*.name", "data.cat.*.color"],
        "input_data": {
            "cat": {
                "cat1": CocoProto,
                "cat2": BoboProto,
            }
        },
        "expected_input": {
            "data": {
                "cat": {
                    "cat1": {"name": "Coco", "color": "calico"},
                    "cat2": {"name": "Bobo", "color": "tuxedo"},
                }
            }
        },
    },
]

# Add to existing test cases section
OUTPUT_FIELDS_TEST_CASES = [
    {
        "name": "dict_output_direct_field",
        "output_fields": ["name", "color"],  # Using dot to reference from root
        "return_value": {"name": "Coco", "color": "calico", "extra": "ignored"},
        "expected_output": {
            "name": "Coco",
            "color": "calico",
        },
    },
    {
        "name": "list_output_array_access",
        "output_fields": ["[*].name"],  # Array access from root
        "return_value": [
            {"name": "Coco", "extra": "ignored"},
            {"name": "Bobo", "extra": "ignored"},
        ],
        "expected_output": [
            {"name": "Coco"},
            {"name": "Bobo"},
        ],
    },
    {
        "name": "protobuf_output_direct_field",
        "output_fields": ["name", "color"],  # Direct field access from root
        "return_value": CocoProto,
        "expected_output": {
            "name": "Coco",
            "color": "calico",
        },
    },
    {
        "name": "dataclass_output_nested_field",
        "output_fields": ["activities[*].name"],  # Nested field from root
        "return_value": CocoDataclass,
        "expected_output": {
            "activities": [
                {"name": "sunbathing"},
                {"name": "brushing"},
            ]
        },
    },
    {
        "name": "direct_field_access",
        "output_fields": ["name"],  # Direct field access
        "return_value": {"name": "Coco", "extra": "ignored"},
        "expected_output": {"name": "Coco"},
    },
]


class TestSnapshotFieldSelection:
    def setup_method(self):
        """Setup before each test."""
        setup_debug_dir()
        os.environ["DEBUG"] = "true"

    @pytest.mark.parametrize(
        "test_case",
        SINGLE_INPUT_TEST_CASES,
        ids=[case["name"] for case in SINGLE_INPUT_TEST_CASES],
    )
    @patch("detective.snapshot._generate_short_hash")
    def test_field_selection_dict(self, mock_hash, test_case):
        """Test field selection patterns with dictionary input."""
        mock_hash.return_value = get_test_hash()

        @snapshot(input_fields=test_case["input_fields"])
        def func(cats: dict) -> bool:
            return True

        # Run the function with CatData_dict
        assert func(CatData_dict)

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": test_case["expected_input"],
            "OUTPUT": True,
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @pytest.mark.parametrize(
        "test_case",
        ARGS_TEST_CASES,
        ids=[case["name"] for case in ARGS_TEST_CASES],
    )
    @patch("detective.snapshot._generate_short_hash")
    def test_field_selection_args(self, mock_hash, test_case):
        """Test field selection patterns with args/kwargs."""
        mock_hash.return_value = get_test_hash()

        @snapshot(input_fields=test_case["input_fields"])
        def func(*args, **kwargs) -> bool:
            return True

        args = test_case.get("args", [])
        kwargs = test_case.get("kwargs", {})
        assert func(*args, **kwargs)

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": test_case["expected_input"],
            "OUTPUT": True,
        }
        assert are_snapshots_equal(actual_data, expected_data)

    # Add new test method for array cases
    @pytest.mark.parametrize(
        "test_case",
        ARRAY_TEST_CASES,
        ids=[case["name"] for case in ARRAY_TEST_CASES],
    )
    @patch("detective.snapshot._generate_short_hash")
    def test_field_selection_arrays(self, mock_hash, test_case):
        """Test field selection patterns with array inputs."""
        mock_hash.return_value = get_test_hash()

        @snapshot(input_fields=test_case["input_fields"])
        def func(cats: List[Any]) -> bool:
            return True

        # Run the function with the test input data
        assert func(test_case["input_data"])

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": test_case["expected_input"],
            "OUTPUT": True,
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @pytest.mark.parametrize(
        "test_case",
        MIXED_OBJECTS_TEST_CASES,
        ids=[case["name"] for case in MIXED_OBJECTS_TEST_CASES],
    )
    @patch("detective.snapshot._generate_short_hash")
    def test_field_selection_mixed_objects(self, mock_hash, test_case):
        """Test field selection patterns with mixed object types."""
        mock_hash.return_value = get_test_hash()

        @snapshot(input_fields=test_case["input_fields"])
        def func(data: Any) -> bool:
            return True

        # Run the function with the test input data
        assert func(test_case["input_data"])

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": test_case["expected_input"],
            "OUTPUT": True,
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @pytest.mark.parametrize(
        "test_case",
        OUTPUT_FIELDS_TEST_CASES,
        ids=[case["name"] for case in OUTPUT_FIELDS_TEST_CASES],
    )
    @patch("detective.snapshot._generate_short_hash")
    def test_field_selection_outputs(self, mock_hash, test_case):
        """Test field selection patterns for function outputs."""
        mock_hash.return_value = get_test_hash()

        @snapshot(output_fields=test_case["output_fields"])
        def func() -> Any:
            return test_case["return_value"]

        # Run the function with the test input data
        result = func()
        assert result == test_case["return_value"]  # Original function output unchanged

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": {},
            "OUTPUT": test_case["expected_output"],
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_instance_method_without_self(self, mock_hash):
        """Test that 'self' is not included when not in input_fields."""
        mock_hash.return_value = get_test_hash()

        class MyClass:
            def __init__(self, a):
                self.a = a

            def instance_method(self, x, y):
                return x + y + self.a

        instance = MyClass(5)

        @snapshot(input_fields=["x"])
        def func(instance, x, y):
            return instance.instance_method(x, y)

        assert func(instance, 10, 20) == 35

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": {"x": 10},
            "OUTPUT": 35,
        }
        assert are_snapshots_equal(actual_data, expected_data)

    @patch("detective.snapshot._generate_short_hash")
    def test_class_method_without_cls(self, mock_hash):
        """Test that 'cls' is not included when not in input_fields."""
        mock_hash.return_value = get_test_hash()

        class MyClass:
            class_variable = 10

            @classmethod
            def class_method(cls, x, y):
                return x + y + cls.class_variable

        @snapshot(input_fields=["y"])
        def func(x, y):
            return MyClass.class_method(x, y)

        assert func(5, 15) == 30

        _, actual_data = get_debug_file(get_test_hash())
        expected_data = {
            "FUNCTION": "func",
            "INPUTS": {"y": 15},
            "OUTPUT": 30,
        }
        assert are_snapshots_equal(actual_data, expected_data)
