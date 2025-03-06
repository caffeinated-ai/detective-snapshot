# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.15] - 2024-03-06
### Added
- Support for SymPy `Symbol` objects as dictionary keys and values.
- Enhanced JSON serialization error handling to provide more informative debugging output when non-serializable objects are encountered.  The `_debug_json_data` function now provides detailed information about the location and type of problematic data within nested structures.
- Improved handling of various edge cases related to field selection and nested objects, leading to more robust and reliable snapshot generation.

### Fixed
- Resolved several regressions introduced by code streamlining in previous versions. Specifically:
    - Reintroduced the `_preprocess_data` function to correctly handle SymPy objects and complex nested structures before JSON serialization. This fixes issues where snapshots involving SymPy were not being generated correctly.
    - Restored the array handling logic in `_process_output_fields` to ensure that selecting specific elements from arrays in the output works as expected. This fixes tests that rely on array indexing in output fields.
    - Added a safety check in `_construct_final_output` to handle potential edge cases (though unlikely) where `inner_calls` might be empty.
- Fixed minor issues in `CustomJSONEncoder` and other helper functions to improve overall code quality and maintainability.

### Changed
- Streamlined and optimized several parts of the `snapshot.py` code for improved readability and efficiency, while ensuring full test coverage and backwards compatibility.  These changes include:
    - Simplification of `CustomJSONEncoder`.
    - Minor improvements to `_extract_and_format_fields` and related functions.
    - Optimization of `_prepare_call_data`.

## [0.1.13] - 2025-02-24

### Fixed
- Added jsbeautifier as required dependency
- Critical reliability improvements to ensure `@snapshot` never throws exceptions that affect the original function

### Added
- New `include_implicit` parameter to `@snapshot()` decorator to control whether 'self' or 'cls' are included in captured inputs. Default is False.

## [0.1.8] - 2025-02-21

### Fixed
- Fixed session handling between calls to the same function.

### Added
- Added support for class, instance, static methods
- Better exception handling
- Support for DEBUG or DETECTIVE environment variables set to TRUE|True|true|1
- Changed snapshot file naming to use timestamps instead of UUIDs

### Changed
- Updated README with more detailed instructions.

## [0.1.1] - 2025-02-20

### Added
- Initial release of detective-snapshot.
- Implemented core snapshot functionality.
- Added support for field selection.
- Added tests for basic functionality.
