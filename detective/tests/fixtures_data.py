import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from google.protobuf.json_format import Parse

from detective.tests.proto.cat_pb2 import Cat as CatProto

logger = logging.getLogger(__name__)


# Test Data
CatData_dict = {
    "Coco": {
        "name": "Coco",
        "color": "calico",
        "foods": ["sushi", "salmon", "tuna"],
        "activities": [
            {"name": "sunbathing", "cuteness": "purrfectly_toasty"},
            {"name": "brushing", "adorableness": "melts_like_butter"},
        ],
    },
    "Bobo": {
        "name": "Bobo",
        "color": "tuxedo",
        "foods": ["kibble", "chicken", "tuna pate"],
        "activities": [{"name": "belly rubs", "goofiness": "rolls_around_happily"}],
    },
    "Jagger": {
        "name": "Jagger",
        "color": "void",
        "foods": ["avocado", "cheese", "salmon treats"],
        "activities": [
            {"name": "shadow prowling", "stealth": "ninja_level"},
            {"name": "shoulder rides", "friendliness": "human_scarf"},
        ],
    },
}

CocoCat = CatData_dict["Coco"]
BoboCat = CatData_dict["Bobo"]
JaggerCat = CatData_dict["Jagger"]


# Dataclass version
@dataclass
class Activity:
    name: str
    cuteness: Optional[str] = None
    adorableness: Optional[str] = None
    goofiness: Optional[str] = None
    stealth: Optional[str] = None
    friendliness: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class Cat:
    name: str
    color: str
    foods: List[str]
    activities: List[Activity]

    def to_dict(self) -> dict:
        """Convert to dict with activities properly handled."""
        return {
            "name": self.name,
            "color": self.color,
            "foods": self.foods,
            "activities": [a.to_dict() for a in self.activities],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Cat":
        return cls(
            name=data["name"],
            color=data["color"],
            foods=data["foods"],
            activities=[
                Activity(
                    name=a["name"],
                    cuteness=a.get("cuteness"),
                    adorableness=a.get("adorableness"),
                    goofiness=a.get("goofiness"),
                    stealth=a.get("stealth"),
                    friendliness=a.get("friendliness"),
                )
                for a in data["activities"]
            ],
        )


# Create dataclass instances for each cat
CocoDataclass = Cat.from_dict(CocoCat)
BoboDataclass = Cat.from_dict(BoboCat)
JaggerDataclass = Cat.from_dict(JaggerCat)

# Keep the old name for backwards compatibility
CatData_dataclass = (
    CocoDataclass  # This was the original instance that used Coco's data
)


def create_cat_proto(cat_data: dict) -> Any:
    """Create a protobuf Cat instance from cat data."""
    message = CatProto()
    Parse(json.dumps(cat_data), message)
    return message


# Create protobuf instances for each cat
CocoProto = create_cat_proto(CatData_dict["Coco"])
BoboProto = create_cat_proto(CatData_dict["Bobo"])
JaggerProto = create_cat_proto(CatData_dict["Jagger"])
