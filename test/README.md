# Test Suite for LLM Topic Modeller

This test suite provides comprehensive testing for the CLI interface (`cli_topics.py`) and GUI application (`app.py`).

## Overview

The test suite includes:
- **CLI Tests**: Tests based on examples from the `cli_topics.py` epilog
- **GUI Tests**: Tests to verify the Gradio interface loads correctly
- **Mock Inference Server**: A dummy inference-server endpoint that avoids API costs during testing

## Structure

- `test.py`: Main test suite with CLI tests
- `test_gui_only.py`: GUI-specific tests
- `mock_inference_server.py`: Mock HTTP server that mimics an inference-server API
- `run_tests.py`: Test runner script
- `__init__.py`: Package initialization

## Running Tests

### Run All Tests

From the project root directory:

```bash
python test/run_tests.py
```

Or from the test directory:

```bash
python run_tests.py
```

### Run Only CLI Tests

```bash
python -m unittest test.test.TestCLITopicsExamples
```

### Run Only GUI Tests

```bash
python test/test_gui_only.py
```

## Mock Inference Server

The test suite uses a mock inference server to avoid API costs during testing. The mock server:

- Listens on `localhost:8080` by default
- Responds to `/v1/chat/completions` endpoint
- Returns valid markdown table responses that satisfy validation requirements
- Provides token counts for usage tracking

The mock server is automatically started before tests and stopped after tests complete.

## Test Coverage

The CLI tests cover:

1. **Topic Extraction**
   - Default settings
   - Custom model and context
   - Grouping by column
   - Zero-shot extraction with candidate topics

2. **Topic Deduplication**
   - Fuzzy matching
   - LLM-based deduplication

3. **All-in-One Pipeline**
   - Complete workflow (extract, deduplicate, summarise)

## Requirements

- Python 3.7+
- All dependencies from `requirements.txt`
- Example data files in `example_data/` directory

## Notes

- Tests will be skipped if required example files are not found
- The mock server must be running for CLI tests to work
- Tests use temporary output directories that are cleaned up after execution

