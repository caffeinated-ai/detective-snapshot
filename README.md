# Detective Snapshot 🕵️‍♂️🔍

A beautifully simple Python debugging tool that automatically logs function inputs, outputs, and exceptions into a clean JSON file. When your code works perfectly nine out of ten times but mysteriously fails on that seventh run, you can see exactly what changed—without starting yet another debugging session.

Just add `@snapshot()` and set DEBUG=true. That’s it! Each time the decorated function is run, detective-snapshot creates a clear, searchable JSON file of everything that went in and what came out—no setup hassles or terminal noise.

You can even select just the parts of the input (including portions of self or cls) you care about using JSON path expressions, keeping your snapshots as focused or comprehensive as you like. The file also captures every call your outer function makes to any decorated inner function, no matter how deeply nested. This has been a total game changer for pinpointing those elusive bugs in complex function chains, and the JSON logs are super easy to drop into an LLM chat for extra debugging help.

Perfect for:
- Catching that one input that led to an unexpected output
- Tracking data transformations across your function chain
- Having a permanent, searchable record of function behavior
- Debugging nested function calls without print statement noise
- Understanding exactly what changed between good and bad runs

## Features
- 📸 Capture function inputs, outputs
- 🌳 Track nested function calls
- 🎯 Select specific fields to snapshot
- 📦 Support for Python objects, dataclasses, and protobufs
- 💥 Capture exception details

## Installation

```bash
pip install detective-snapshot
```

## Quick Start

Enable debug mode by setting either environment variable:
```bash
export DEBUG=true
# or
export DETECTIVE=true
```

With debug mode on, each call to an outermost decorated function creates a new snapshot file under `./_snapshots/` with a timestamp and unique hash.

Here's a simple example using a library catalog system:

```python
from detective import snapshot

@snapshot()
def get_book_details(book):
    author = get_author(book["author_id"])
    return f"{book['title']} by {author}"

@snapshot()
def get_author(author_id):
    # Simulate database lookup
    return "J.K. Rowling"

# Use the functions
book = {
    "title": "Harry Potter",
    "author_id": "jkr_001"
}
result = get_book_details(book)
```

This will create a debug file in `./_snapshots/` with content like:

```json
{
    "FUNCTION": "get_book_details",
    "INPUTS": {
        "book": {
            "title": "Harry Potter",
            "author_id": "jkr_001"
        }
    },
    "OUTPUT": "Harry Potter by J.K. Rowling",
    "CALLS": [
        {
            "FUNCTION": "get_author",
            "INPUTS": {
                "author_id": "jkr_001"
            },
            "OUTPUT": "J.K. Rowling"
        }
    ]
}
```

## Field Selection

Detective Snapshot supports both its own simple field selection syntax and full [JSONPath](https://github.com/h2non/jsonpath-ng) expressions out of the box. You can capture specific fields using various selection patterns:

```python
@snapshot(
    input_fields=["book.title", "book.author_id"],
    output_fields=["name"]
)
def process_book(book):
    # Only specified fields will be captured
    pass
```

### Supported Field Selection Patterns

| Pattern | Example | Description |
|---------|---------|-------------|
| Direct Field | `name` | Select a field directly from root |
| Nested Field | `user.address.city` | Navigate through nested objects |
| Array Index | `books[0].title` | Select specific array element |
| Array Wildcard | `books[*].title` | Select field from all array elements |
| Multiple Fields | `user.(name,age)` | Select multiple fields from an object |
| Wildcard Object | `users.*.name` | Select field from all child objects |
| Args Syntax | `args[0].name` | Select from function arguments |
| Mixed Access | `users[*].addresses.*.city` | Combine array and object access |
| JSONPath | `$.users[?(@.age > 18)].name` | Use full JSONPath expressions |

For more examples of field selection patterns, check out our test files - particularly `test_snapshot_fields_selection.py` which contains comprehensive examples of different selection patterns and edge cases.

## Advanced Usage

### Capture Complex Objects

```python
@dataclass
class Book:
    title: str
    author: str
    chapters: List[Chapter]

@snapshot(input_fields=["book.chapters[*].title"])
def get_chapter_titles(book: Book):
    return [chapter.title for chapter in book.chapters]
```

### Handle Nested Function Calls

```python
@snapshot()
def process_library(library):
    books = get_books(library.id)
    return categorize_books(books)

@snapshot()
def get_books(library_id):
    return ["Book1", "Book2"]

@snapshot()
def categorize_books(books):
    return {"fiction": books}
```

The debug file will include the complete call hierarchy with inputs and outputs for each function.

### Exception Handling

If an exception occurs within a function decorated with `@snapshot`, Detective Snapshot will capture the exception details.  The output will include an `error` field containing the exception type and message.  Nested function calls that *also* raise exceptions will have their exceptions captured within the `CALLS` section of the parent function.

```python
@snapshot()
def outer_function():
    try:
        inner_function()
    except ValueError:
        pass

@snapshot()
def inner_function():
    raise ValueError("This is an example error.")

outer_function()
```

This will produce a snapshot similar to:

```json
{
    "FUNCTION": "outer_function",
    "INPUTS": {},
    "OUTPUT": null, // or {} depending on if there's a return before the exception
    "CALLS": [
        {
            "FUNCTION": "inner_function",
            "INPUTS": {},
            "OUTPUT": {
                "error": {
                    "type": "ValueError",
                    "message": "This is an example error."
                }
            }
        }
    ]
}
```
If the outermost function raises the exception, the `OUTPUT` will contain the error.

### Class, Instance, and Static Methods

Detective Snapshot works seamlessly with class methods, instance methods, and static methods.

*   **Class Methods:** The `cls` parameter (the class itself) will be captured in the `INPUTS`. Only non-internal, non-callable, and non-decorator attributes of the class are included.
*   **Instance Methods:** The `self` parameter (the instance) will be captured.  If `self` has a `__dict__` attribute, it will be captured; otherwise, Detective Snapshot attempts to serialize it.
*   **Static Methods:** Static methods are treated like regular functions.

Here's an example demonstrating all three:

```python
class MyClass:
    class_variable = "I'm a class variable!"

    def __init__(self, value):
        self.instance_variable = value

    @snapshot()
    def instance_method(self, x):
        return self.static_method(x + self.instance_variable)

    @classmethod
    @snapshot()
    def class_method(cls, y):
        return y * 2

    @staticmethod
    @snapshot()
    def static_method(z):
        return z * 3

instance = MyClass(10)
instance.instance_method(5)
MyClass.class_method(8)

```
The resulting snapshots will capture the relevant `self`, `cls`, and other parameters for each method type.

## Contributing

Contributions are welcome! Please check out our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.