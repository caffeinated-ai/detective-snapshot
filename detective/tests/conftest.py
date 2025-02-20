import glob
import os
import pytest
from detective.snapshot import inner_calls_var, session_id_var


def pytest_addoption(parser):
    parser.addoption(
        "--update",
        action="store_true",
        default=False,
        help="Update the expected JSON files with new snapshots",
    )


@pytest.fixture
def update_snapshots(request):
    return request.config.getoption("--update")


@pytest.fixture(autouse=True)
def cleanup_snapshot_files(request):
    """Cleanup snapshot files after each test, unless the test failed."""
    yield  # Run the test

    # Get the test name from the request object
    test_name = request.node.name

    # Only cleanup if test passed
    if request.node.session.testsfailed == 0:
        # Remove all JSON files in debug_snapshots directory
        files = glob.glob("debug_snapshots/*.json")
        for file in files:
            try:
                os.remove(file)
            except OSError:
                pass
    else:
        print(f"Test {test_name} failed - preserving snapshot files for debugging")


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Reset context variables before each test."""
    inner_calls_var.set([])
    session_id_var.set(None)
    yield
