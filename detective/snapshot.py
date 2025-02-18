import functools
import inspect
import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from detective.proto_utils import protobuf_to_json
from jsonpath_ng.ext import parse as parse_ext

logger = logging.getLogger(__name__)

# --- Context Variables for Session Management ---
session_id_var: ContextVar[Optional[str]] = ContextVar("debug_session_id", default=None)
inner_calls_var: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "inner_function_calls", default=None
)


class PathComponent:
    """Represents a component of a field path."""

    def __init__(
        self,
        name: str,
        array_index: Optional[Union[int, str]] = None,
        fields: Optional[List[str]] = None,
    ):
        self.name = name
        self.array_index = array_index
        self.fields = fields

    @property
    def is_wildcard(self) -> bool:
        return self.array_index == "*"

    @property
    def is_array_access(self) -> bool:
        return self.array_index is not None

    @property
    def has_fields(self) -> bool:
        return self.fields is not None


def _parse_path(field_path: str) -> List[PathComponent]:
    """
    Parse a field path into components.

    Args:
        field_path: Path expression (e.g., 'items[0].name', 'items[*].value')

    Returns:
        List of PathComponent objects
    """
    if "(" in field_path and ")" in field_path:
        # Handle multiple field selection syntax
        base_path, fields = field_path.split("(", 1)
        fields = [f.strip() for f in fields.rstrip(")").split(",")]

        # Check if base path has array access
        if "[*]" in base_path:
            base_name = base_path[: base_path.index("[")]
            return [PathComponent(base_name, array_index="*", fields=fields)]
        return [PathComponent(base_path.rstrip("."), fields=fields)]

    components = []
    current = ""
    i = 0
    while i < len(field_path):
        char = field_path[i]
        if char == "[":
            if current:
                # Start of array access
                j = i + 1
                while j < len(field_path) and field_path[j] != "]":
                    j += 1
                if j < len(field_path):
                    index = field_path[i + 1 : j]
                    array_index: Union[int, str] = index if index == "*" else int(index)
                    components.append(PathComponent(current, array_index=array_index))
                    i = j + 1
                    current = ""
                else:
                    raise ValueError(f"Invalid array syntax in path: {field_path}")
        elif char == ".":
            if current:
                components.append(PathComponent(current))
                current = ""
        else:
            current += char
        i += 1

    if current:
        components.append(PathComponent(current))

    return components


def _convert_path(field_path: str) -> str:
    """
    Convert our simplified path syntax to jsonpath-ng syntax.

    Args:
        field_path: Path in simplified format (e.g., 'items[0].name')

    Returns:
        Path in jsonpath-ng format
    """
    # Handle multiple field selection
    if "(" in field_path and ")" in field_path:
        base_path, fields = field_path.split("(", 1)
        fields = [f.strip() for f in fields.rstrip(")").split(",")]
        # Convert to jsonpath-ng multi-field syntax
        return f"{base_path}[*].{{{','.join(fields)}}}"

    # Convert array access syntax
    # Replace [0] with [0] and [*] with [*]
    parts = []
    current = ""
    i = 0
    while i < len(field_path):
        if field_path[i] == "[":
            if current:
                parts.append(current)
                current = ""
            # Find matching bracket
            j = i + 1
            while j < len(field_path) and field_path[j] != "]":
                j += 1
            if j < len(field_path):
                index = field_path[i + 1 : j]
                parts.append(f"[{index}]")
                i = j + 1
            else:
                raise ValueError(f"Invalid array syntax in path: {field_path}")
        elif field_path[i] == ".":
            if current:
                parts.append(current)
                current = ""
        else:
            current += field_path[i]
        i += 1

    if current:
        parts.append(current)

    return ".".join(parts)


def _get_nested_value(obj: Any, field_path: str) -> Any:
    """
    Get a nested value from an object using a simplified path syntax.

    Args:
        obj: The object to extract value from
        field_path: Path expression. Supports:
            - Direct field access: 'result_items'
            - Dot notation: 'user.profile.name'
            - Array indexing: 'items[0].id'
            - Wildcard array: 'items[*].name'
            - Multiple fields: 'items[*].(name,price)'

    Returns:
        The value(s) matching the path expression, or None if not found
    """
    try:
        # Handle kwargs special case for function parameter access
        if isinstance(obj, dict) and "kwargs" in obj:
            obj = obj.get("kwargs", {})

        logger.debug(f"Searching with path: {field_path}")
        logger.debug(f"Object type: {type(obj)}")
        logger.debug(f"Object content: {obj}")

        # Parse the path into components
        components = _parse_path(field_path)
        logger.debug(f"Path components: {components}")

        # Handle multiple field selection
        if components and components[0].has_fields:
            component = components[0]
            # Get the array value first
            array_value = None
            if isinstance(obj, dict):
                array_value = obj.get(component.name)
            else:
                array_value = getattr(obj, component.name, None)

            if not isinstance(array_value, (list, tuple)):
                return None

            # Extract specified fields from each item
            result = []
            for item in array_value:
                if item is None:
                    result.append(None)
                    continue
                extracted = {}
                for field in component.fields or []:  # Handle None fields
                    field = field.strip()  # Ensure no whitespace
                    if isinstance(item, dict):
                        extracted[field] = item.get(field)
                    else:
                        extracted[field] = getattr(item, field, None)
                result.append(extracted)
            return result

        # Process each path component
        current_value = obj
        for i, component in enumerate(components):
            if current_value is None:
                return None

            # Get the value for this component
            if isinstance(current_value, dict):
                current_value = current_value.get(component.name)
            else:
                current_value = getattr(current_value, component.name, None)

            # Handle array access if needed
            if component.is_array_access:
                if not isinstance(current_value, (list, tuple)):
                    return None

                if component.is_wildcard:
                    # For wildcards, process remaining path for each item
                    remaining_components = components[i + 1 :]
                    if not remaining_components:
                        # If this is the last component, return all values
                        return current_value

                    # Process remaining path for each item
                    results = []
                    for item in current_value:
                        if item is None:
                            results.append(None)
                            continue
                        # Recursively process remaining path
                        remaining_path = ".".join(
                            c.name + (f"[{c.array_index}]" if c.is_array_access else "")
                            for c in remaining_components
                        )
                        result = _get_nested_value(item, remaining_path)
                        results.append(result)
                    return results
                else:
                    # Handle specific index access
                    try:
                        current_value = current_value[component.array_index]  # type: ignore
                    except (IndexError, TypeError):
                        return None

        return current_value

    except Exception as e:
        logger.error(f"Error evaluating path '{field_path}': {e}")
        return None


def _to_json_compatible(obj: Any) -> Any:
    """Convert any object to a JSON-compatible format."""
    if obj is None:
        return None

    # Handle protobuf objects
    if hasattr(obj, "DESCRIPTOR") or hasattr(obj, "_pb"):
        try:
            json_str = protobuf_to_json(obj)
            return json.loads(json_str)
        except Exception:
            # If protobuf_to_json fails, try to get the value attribute
            if hasattr(obj, "value"):
                return _to_json_compatible(obj.value)
            return str(obj)

    # Handle dataclasses
    if hasattr(obj, "__dataclass_fields__"):
        return {
            field: _to_json_compatible(getattr(obj, field))
            for field in obj.__dataclass_fields__
        }

    # Handle enums
    if isinstance(obj, Enum):
        return obj.value

    # Handle objects with to_dict method
    if hasattr(obj, "to_dict"):
        return _to_json_compatible(obj.to_dict())

    # Handle objects with __dict__
    if hasattr(obj, "__dict__"):
        return _to_json_compatible(obj.__dict__)

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_to_json_compatible(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: _to_json_compatible(value) for key, value in obj.items()}

    # Basic types (str, int, float, bool) are already JSON-compatible
    return obj


def _extract_fields(
    obj: Any, field_paths: List[str]
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Extract specified fields using jsonpath-ng expressions.
    First converts the object to a JSON-compatible format.

    Args:
        obj: Object to extract fields from
        field_paths: List of path expressions to extract. Supports JSONPath syntax plus:
            - Direct field access: 'result_items'
            - Dot notation: 'user.profile.name'
            - Array indexing: 'items[0].id'
            - Wildcard array: 'items[*].name'
            - Multiple fields: 'items[*].(field1, field2)'

    Returns:
        Dictionary with extracted fields
    """
    if obj is None:
        return {}

    # Convert object to JSON-compatible format
    json_obj = _to_json_compatible(obj)

    # Handle kwargs special case for function parameter access
    if isinstance(json_obj, dict) and "kwargs" in json_obj:
        json_obj = json_obj.get("kwargs", {})

    result = {}
    for path in field_paths:
        try:
            # Handle multiple field selection with parentheses
            if "(" in path and ")" in path:
                base_path, fields = path.split("(", 1)
                # Clean up fields string and split by comma, handling spaces
                fields = [f.strip() for f in fields.rstrip(")").split(",")]
                # Remove any empty strings that might result from extra spaces
                fields = [f for f in fields if f]

                # Convert base path to JSONPath
                if not base_path.startswith("$"):
                    base_path = "$." + base_path

                # Get base object(s) using parse_ext for better array handling
                base_expr = parse_ext(base_path)
                base_matches = base_expr.find(json_obj)

                if not base_matches:
                    continue

                # Extract each field for each match
                extracted = []
                for match in base_matches:
                    item = {}
                    for field in fields:
                        try:
                            # Handle nested fields using parse_ext
                            field_expr = parse_ext(f"$.{field}")
                            field_matches = field_expr.find(match.value)
                            if field_matches:
                                item[field] = field_matches[0].value
                            else:
                                # Try direct attribute/dict access as fallback
                                if isinstance(match.value, dict):
                                    item[field] = match.value.get(field)
                                else:
                                    item[field] = getattr(match.value, field, None)
                        except Exception as e:
                            logger.debug(f"Error extracting field '{field}': {e}")
                            item[field] = None
                    extracted.append(item)

                # Store result with original path (without $)
                result[path] = extracted[0] if len(extracted) == 1 else extracted
            else:
                # Convert to JSONPath if needed
                if not path.startswith("$"):
                    jsonpath = "$." + path
                else:
                    jsonpath = path
                    path = (
                        path[2:] if path.startswith("$.") else path
                    )  # Remove $. prefix

                # Use extended parser for better array handling
                expr = parse_ext(jsonpath)
                matches = expr.find(json_obj)

                if matches:
                    values = [match.value for match in matches]
                    result[path] = values[0] if len(values) == 1 else values

        except Exception as e:
            logger.error(f"Error extracting path '{path}': {e}")
            continue

    return result


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dataclasses, protobuf objects, and enums."""

    def default(self, obj):
        # Handle protobuf objects
        if hasattr(obj, "DESCRIPTOR") or hasattr(obj, "_pb"):
            try:
                json_str = protobuf_to_json(obj)
                return json.loads(json_str)
            except Exception:
                # If protobuf_to_json fails, try to get the value attribute
                if hasattr(obj, "value"):
                    return self.default(obj.value)
                return str(obj)

        # Handle dataclasses
        if hasattr(obj, "__dataclass_fields__"):
            return {field: getattr(obj, field) for field in obj.__dataclass_fields__}

        # Handle enums
        if isinstance(obj, Enum):
            return obj.value

        # Handle objects with to_dict method
        if hasattr(obj, "to_dict"):
            return obj.to_dict()

        # Handle objects with __dict__
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        # Let the base class handle the rest
        return super().default(obj)


def snapshot(input_fields=None, output_fields=None):
    """
    Decorator that logs inputs & outputs to a single debug file,
    reading & writing session_id from a ContextVar.  Also handles
    nested function calls.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- Session ID Handling ---
            session_id = session_id_var.get()
            if session_id is None:
                session_id = str(uuid.uuid4())
                session_id_var.set(session_id)

            # --- Inner Call Tracking ---
            is_outermost = False
            inner_calls = inner_calls_var.get()
            if inner_calls is None:
                is_outermost = True  # This is the top-level call
                inner_calls = []
                inner_calls_var.set(inner_calls)

            # --- Convert Args to Kwargs (for consistent field access) ---
            try:
                if isinstance(func, staticmethod):
                    actual_func = func.__get__(None, type(None))
                else:
                    actual_func = func
                sig = inspect.signature(actual_func)
                param_names = list(sig.parameters.keys())
                if param_names and param_names[0] in ("self", "cls"):
                    param_names = param_names[1:]
                # Create a new kwargs dict with positional args
                all_kwargs = dict(zip(param_names, args))
                # Update with provided kwargs
                all_kwargs.update(kwargs)
            except Exception as e:
                logger.warning(f"Failed to get function parameters: {e}")
                all_kwargs = kwargs

            # --- Capture Input (before calling the function) ---
            current_call_data = {
                "function_name": func.__name__,
                "input": {"kwargs": all_kwargs},  # Store kwargs directly
                "output": None,  # Placeholder
            }

            # --- Extract Input Fields (if specified) ---
            extracted_input = None
            if input_fields:
                extracted_input = _extract_fields(
                    current_call_data["input"], input_fields
                )

            # --- Call the Function and Capture Output---
            try:
                result = func(*args, **kwargs)  # Use original args and kwargs
                current_call_data["output"] = result  # Store raw result
            except Exception as e:
                current_call_data["output"] = {"error": str(e)}  # Capture exceptions
                # --- Add to Inner Calls (if not outermost) ---
                if not is_outermost:
                    inner_calls.insert(0, current_call_data)  # Insert at beginning
                # --- Debug Mode and File Writing (outermost only) ---
                if (
                    os.getenv("DEBUG", "").lower() in ("1", "true", "yes", "on")
                    and is_outermost
                ):
                    debug_dir = os.path.join(os.getcwd(), "debug_output")
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{session_id}_{timestamp}.json"  # Simpler filename
                    filepath = os.path.join(debug_dir, filename)

                    # --- Prepare Final Output (outermost function) ---
                    final_output = {
                        "input": (
                            extracted_input
                            if extracted_input
                            else current_call_data["input"]
                        ),
                        "output": current_call_data["output"],
                        "inner_function_calls": inner_calls,
                    }

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(
                            final_output,
                            f,
                            indent=2,
                            ensure_ascii=False,
                            cls=CustomJSONEncoder,
                        )
                    logger.debug(f"Debug data written to {filepath}")

                    # --- Reset inner_calls_var for next top-level call ---
                    inner_calls_var.set(None)
                raise  # Re-raise the exception

            # --- Extract Output Fields (if specified) ---
            extracted_output = None
            if output_fields:
                if is_outermost:
                    # For outermost function, use the standard extraction
                    extracted_output = _extract_fields(
                        {"result": current_call_data["output"]}, output_fields
                    )
                else:
                    # For inner functions, create the expected output format directly
                    field_name = output_fields[
                        0
                    ]  # We know there's only one field for inner functions
                    extracted_output = {field_name: current_call_data["output"]}

            # --- Add to Inner Calls (if not outermost) ---
            if not is_outermost:
                if extracted_output:
                    current_call_data["output"] = extracted_output
                inner_calls.insert(0, current_call_data)  # Insert at beginning

            # --- Debug Mode and File Writing (outermost only) ---
            if (
                os.getenv("DEBUG", "").lower() in ("1", "true", "yes", "on")
                and is_outermost
            ):
                debug_dir = os.path.join(os.getcwd(), "debug_output")
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{session_id}_{timestamp}.json"  # Simpler filename
                filepath = os.path.join(debug_dir, filename)

                # --- Prepare Final Output (outermost function) ---
                final_output = {
                    "input": (
                        extracted_input
                        if extracted_input
                        else current_call_data["input"]
                    ),
                    "output": (
                        extracted_output
                        if extracted_output
                        else current_call_data["output"]
                    ),
                    "inner_function_calls": inner_calls,
                }

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(
                        final_output,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        cls=CustomJSONEncoder,
                    )
                logger.debug(f"Debug data written to {filepath}")

                # --- Reset inner_calls_var for next top-level call ---
                inner_calls_var.set(None)

            return result

        return wrapper

    return decorator
