# Detective Snapshot 🕵️‍♂️🔍

A Python package for capturing function input/output snapshots. Perfect for debugging, testing, and understanding complex function call hierarchies.

## Features
- 📸 Capture function inputs and outputs
- 🌳 Track nested function calls
- 🎯 Select specific fields to snapshot
- 📦 Support for Python objects, dataclasses, and protobufs

## Install

```bash
pip install detective-snapshot
```

## Quick Start

```python
from detective import snapshot

@snapshot()
def calculate_cat_food(cats):
    total = sum(get_cat_appetite(cat) for cat in cats)
    return f"Need {total}oz of food per day"

@snapshot()
def get_cat_appetite(cat):
    # Cats eat about 0.5oz per pound of weight
    return cat["weight"] * 0.5

# Feed the kitties
cats = [
    {"name": "Whiskers", "weight": 10},
    {"name": "Chonk", "weight": 15}
]
result = calculate_cat_food(cats)
```

With debug mode on, each call to an outermost decorated function creates a new snapshot file in `./debug_snapshots/` with a unique UUID:

```json
{
    "FUNCTION": "calculate_cat_food",
    "INPUTS": {
        "cats": [
            {"name": "Whiskers", "weight": 10},
            {"name": "Chonk", "weight": 15}
        ]
    },
    "OUTPUT": "Need 12.5oz of food per day",
    "CALLS": [
        {
            "FUNCTION": "get_cat_appetite",
            "INPUTS": {"cat": {"name": "Whiskers", "weight": 10}},
            "OUTPUT": 5.0
        },
        {
            "FUNCTION": "get_cat_appetite",
            "INPUTS": {"cat": {"name": "Chonk", "weight": 15}},
            "OUTPUT": 7.5
        }
    ]
}
```

## Field Selection

Capture only the fields you care about:

```python
@snapshot(
    input_fields=["cat.name", "cat.weight"],
    output_fields=["daily_food"]
)
def calculate_cat_diet(cat):
    daily_food = cat["weight"] * 0.5
    return {"daily_food": daily_food, "feeding_schedule": "twice daily"}
```

### Selection Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| Direct Field | `name` | Select a field directly from root |
| Nested Field | `cat.weight` | Navigate through nested objects |
| Array Index | `cats[0].name` | Select specific array element |
| Array Wildcard | `cats[*].weight` | Select field from all array elements |
| Multiple Fields | `cat.(name,weight)` | Select multiple fields from an object |
| Wildcard Object | `pets.*.name` | Select field from all child objects |
| Args Syntax | `args[0].name` | Select from function arguments |
| Mixed Access | `pets[*].records.*.weight` | Combine array and object access |

Full [JSONPath](https://github.com/h2non/jsonpath-ng) support is also available for more complex queries.

## Advanced Usage

### Capture Complex Objects

```python
@dataclass
class Cat:
    name: str
    breed: str
    medical_records: List[Record]

@snapshot(input_fields=["cat.medical_records[*].weight"])
def track_weight_history(cat: Cat):
    return [record.weight for record in cat.medical_records]
```

### Handle Nested Function Calls

```python
@snapshot()
def process_cattery(cattery):
    cats = get_cats(cattery.id)
    return categorize_cats(cats)

@snapshot()
def get_cats(cattery_id):
    return ["Whiskers", "Chonk"]

@snapshot()
def categorize_cats(cats):
    return {"chonky": cats}
```

The debug file will include the complete call hierarchy with inputs and outputs for each function.

## Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE)
