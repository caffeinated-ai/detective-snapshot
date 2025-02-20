"""Shared utilities for tests."""

import difflib
import inspect
import json
import os
from typing import Any, Dict, Tuple


def get_debug_file(mock_uuid_str: str) -> Tuple[str, Dict[str, Any]]:
    """Get the debug file path and its contents.

    Args:
        mock_uuid_str: The UUID string to match against filenames.

    Returns:
        Tuple of (file path, file contents as dict)
    """
    debug_dir = os.path.join(os.getcwd(), "debug_snapshots")
    debug_files = [
        f
        for f in os.listdir(debug_dir)
        if f.endswith(".json") and f.startswith(mock_uuid_str)
    ]
    assert len(debug_files) == 1, "Expected exactly one debug file."

    filepath = os.path.join(debug_dir, debug_files[0])
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    return filepath, data


def load_expected_data(test_name: str) -> Dict[str, Any]:
    """Load the expected data for a test.

    Args:
        test_name: Name of the test function

    Returns:
        The expected data as a dictionary
    """
    expected_data_path = get_expected_data_path(test_name)
    with open(expected_data_path) as f:
        return json.load(f)


def get_test_name() -> str:
    """Get the name of the current test function.

    Returns:
        The name of the current test function
    """
    frame = inspect.currentframe()
    try:
        # Go up one frame to get the test function's frame
        test_frame = frame.f_back if frame is not None else None
        return test_frame.f_code.co_name if test_frame is not None else "unknown_test"
    finally:
        # Explicitly delete frame references to help garbage collection
        del frame


def get_expected_data_path(test_name: str) -> str:
    """Get the path to the expected data file.

    Args:
        test_name: Name of the test function

    Returns:
        Path to the expected data file
    """
    return os.path.join(os.path.dirname(__file__), "expected", f"{test_name}.json")


def update_expected_data(actual_filepath: str, test_name: str) -> None:
    """Update the expected data file with actual data.

    Args:
        actual_filepath: Path to the actual data file
        test_name: Name of the test function
    """
    expected_data_path = get_expected_data_path(test_name)
    os.makedirs(os.path.dirname(expected_data_path), exist_ok=True)
    with open(actual_filepath) as source, open(expected_data_path, "w") as target:
        target.write(source.read())
    print(f"\nUpdated expected data: {expected_data_path}")


def are_snapshots_equal(actual: Any, expected: Any) -> bool:
    """Compare two snapshots, handling dictionary key order differences.
    Returns True if snapshots match, False otherwise."""
    if actual != expected:
        # ANSI color codes for better visibility
        RED = "\033[91m"
        GREEN = "\033[92m"
        BLUE = "\033[94m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        # Print full values in one line
        print(f"\n{YELLOW}Actual:   {json.dumps(actual, sort_keys=True)}{RESET}")
        print(f"{BLUE}Expected: {json.dumps(expected, sort_keys=True)}{RESET}")

        print(f"\n{YELLOW}=== Detailed Comparison ==={RESET}")

        # Format JSON with compact representation for better diff readability
        def format_json(obj: Any) -> str:
            return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)

        # Generate and show diff
        diff = difflib.ndiff(
            format_json(actual).splitlines(), format_json(expected).splitlines()
        )

        for line in diff:
            if line.startswith("+"):
                print(f"{GREEN}{line}{RESET}")
            elif line.startswith("-"):
                print(f"{RED}{line}{RESET}")
            elif not line.startswith("?"):  # Skip diff markers
                print(line)
        return False
    return True


def setup_debug_dir() -> None:
    """Setup/cleanup the debug snapshots directory and reset context vars."""
    from detective.snapshot import inner_calls_var, session_id_var

    inner_calls_var.set([])
    session_id_var.set(None)

    debug_dir = os.path.join(os.getcwd(), "debug_snapshots")
    if os.path.exists(debug_dir):
        for file in os.listdir(debug_dir):
            file_path = os.path.join(debug_dir, file)
            try:
                os.remove(file_path)
            except OSError:
                pass
    else:
        os.makedirs(debug_dir)
