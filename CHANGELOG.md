# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
