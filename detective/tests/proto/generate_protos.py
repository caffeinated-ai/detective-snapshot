"""Script to generate Python code from proto files."""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def generate_proto():
    """Generate Python code from proto files."""
    proto_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info("Generating protos in directory: %s", proto_dir)

    proto_file = os.path.join(proto_dir, "cat.proto")
    logger.info("Proto file contents:")
    with open(proto_file) as f:
        logger.info(f.read())

    # Run protoc command
    cmd = [
        "protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={proto_dir}",
        proto_file,
    ]
    logger.info("Running command: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.info("Command output: %s", result.stdout)
    if result.stderr:
        logger.warning("Command stderr: %s", result.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_proto()
