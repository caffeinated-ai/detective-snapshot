import json
import logging
from typing import Type, TypeVar, Union, Any

from google.protobuf.json_format import MessageToDict, Parse
from google.protobuf.message import Message

logger = logging.getLogger(__name__)

# Generic type for protobuf messages or wrappers that must be a subclass of Message
T = TypeVar("T", bound=Union[Message, object])


def protobuf_to_json(proto_obj: Any) -> str:
    """
    Convert a protobuf message to a JSON string.
    Handles objects with ._pb attribute (like VideoAnnotationResults).

    Args:
        proto_obj: The protobuf message to convert

    Returns:
        JSON string representation of the protobuf message

    Raises:
        Exception: If conversion fails
    """
    try:
        # Handle objects that wrap protobuf with ._pb
        if hasattr(proto_obj, "_pb"):
            proto_obj = proto_obj._pb

        # Convert to dict first
        dict_obj = MessageToDict(
            proto_obj,  # type: ignore
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )

        # Convert dict to JSON string
        return json.dumps(dict_obj, indent=4, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error converting protobuf to JSON: {e}")
        raise


def json_to_protobuf(json_str: str, message_type: Type[T]) -> T:
    """
    Convert a JSON string back to a protobuf message.
    For types that wrap protobuf (like VideoAnnotationResults), returns the wrapper.

    Args:
        json_str: The JSON string to convert
        message_type: The target protobuf message type

    Returns:
        Instance of the specified message type

    Raises:
        Exception: If conversion fails
    """
    try:
        # Create a new instance of the message type
        message = message_type()

        # If it's a wrapper type (has ._pb), parse into the internal protobuf
        if hasattr(message, "_pb"):
            Parse(json_str, message._pb)
        else:
            Parse(json_str, message)  # type: ignore

        return message

    except Exception as e:
        logger.error(f"Error converting JSON to protobuf: {e}")
        raise
