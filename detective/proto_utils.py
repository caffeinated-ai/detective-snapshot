from typing import Any, Protocol, TypeVar, Union

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message

# Generic type for protobuf messages or wrappers
T = TypeVar("T", bound=Union[Message, object])


class HasProtobuf(Protocol):
    """Protocol for objects that wrap a protobuf message."""

    _pb: Message


def is_protobuf(obj: Any) -> bool:
    """Check if an object is a protobuf message.

    Args:
        obj: Object to check

    Returns:
        True if the object is a protobuf message, False otherwise
    """
    # Check for standard protobuf DESCRIPTOR attribute
    if hasattr(obj, "DESCRIPTOR"):
        return True

    # Check for _pb attribute (used in some protobuf implementations)
    if hasattr(obj, "_pb") and isinstance(obj._pb, Message):
        return True

    return False


def protobuf_to_dict(proto_obj: Union[Message, HasProtobuf]) -> dict:
    """
    Convert a protobuf message to a dictionary.
    Handles objects with ._pb attribute.

    Args:
        proto_obj: The protobuf message to convert

    Returns:
        Dictionary representation of the protobuf message

    Raises:
        Exception: If conversion fails
    """
    try:
        # Handle objects that wrap protobuf with ._pb
        if hasattr(proto_obj, "_pb"):
            proto_obj = proto_obj._pb  # type: ignore

        # Convert to dict
        return MessageToDict(
            proto_obj,  # type: ignore
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )

    except Exception:
        raise
