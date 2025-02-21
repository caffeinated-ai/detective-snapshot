import glob
import os

import pytest

from detective.snapshot import inner_calls_var, session_id_var, session_start_time_var


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
        # Remove all files in _snapshots directory
        snapshots_dir = os.path.join(os.getcwd(), "_snapshots")
        if os.path.exists(snapshots_dir):
            for root, dirs, files in os.walk(snapshots_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            try:
                os.rmdir(snapshots_dir)
            except OSError:
                pass
    else:
        print(f"Test {test_name} failed - preserving snapshot files for debugging")


@pytest.fixture(autouse=True)
def reset_context_vars():
    """Reset context variables before each test."""
    inner_calls_var.set([])
    session_id_var.set(None)
    session_start_time_var.set(None)
    yield
