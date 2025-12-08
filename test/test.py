import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from typing import List, Optional

# Mock LLM calls are automatically applied via environment variables
# No need to import - the mock patches are applied when USE_MOCK_LLM=1 is set


def run_cli_topics(
    script_path: str,
    task: str,
    output_dir: str,
    input_file: Optional[str] = None,
    text_column: Optional[str] = None,
    previous_output_files: Optional[List[str]] = None,
    timeout: int = 600,  # 10-minute timeout
    # General Arguments
    username: Optional[str] = None,
    save_to_user_folders: Optional[bool] = None,
    excel_sheets: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    # Model Configuration
    model_choice: Optional[str] = None,
    temperature: Optional[float] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    api_url: Optional[str] = None,
    inference_server_model: Optional[str] = None,
    # Topic Extraction Arguments
    context: Optional[str] = None,
    candidate_topics: Optional[str] = None,
    force_zero_shot: Optional[str] = None,
    force_single_topic: Optional[str] = None,
    produce_structured_summary: Optional[str] = None,
    sentiment: Optional[str] = None,
    additional_summary_instructions: Optional[str] = None,
    # Validation Arguments
    additional_validation_issues: Optional[str] = None,
    show_previous_table: Optional[str] = None,
    output_debug_files: Optional[str] = None,
    max_time_for_loop: Optional[int] = None,
    # Deduplication Arguments
    method: Optional[str] = None,
    similarity_threshold: Optional[int] = None,
    merge_sentiment: Optional[str] = None,
    merge_general_topics: Optional[str] = None,
    # Summarisation Arguments
    summary_format: Optional[str] = None,
    sample_reference_table: Optional[str] = None,
    no_of_sampled_summaries: Optional[int] = None,
    random_seed: Optional[int] = None,
    # Output Format Arguments
    create_xlsx_output: Optional[bool] = None,
    # Logging Arguments
    save_logs_to_csv: Optional[bool] = None,
    save_logs_to_dynamodb: Optional[bool] = None,
    cost_code: Optional[str] = None,
) -> bool:
    """
    Executes the cli_topics.py script with specified arguments using a subprocess.

    Args:
        script_path (str): The path to the cli_topics.py script.
        task (str): The main task to perform ('extract', 'validate', 'deduplicate', 'summarise', 'overall_summary', or 'all_in_one').
        output_dir (str): The path to the directory for output files.
        input_file (str, optional): Path to the input file to process.
        text_column (str, optional): Name of the text column to process.
        previous_output_files (List[str], optional): Path(s) to previous output files.
        timeout (int): Timeout in seconds for the subprocess.

        All other arguments match the CLI arguments from cli_topics.py.

    Returns:
        bool: True if the script executed successfully, False otherwise.
    """
    # 1. Get absolute paths and perform pre-checks
    script_abs_path = os.path.abspath(script_path)
    output_abs_dir = os.path.abspath(output_dir)

    # Handle input file based on task
    if task in ["extract", "validate", "all_in_one"] and input_file is None:
        raise ValueError(f"Input file is required for '{task}' task")

    if input_file:
        input_abs_path = os.path.abspath(input_file)
        if not os.path.isfile(input_abs_path):
            raise FileNotFoundError(f"Input file not found: {input_abs_path}")

    if not os.path.isfile(script_abs_path):
        raise FileNotFoundError(f"Script not found: {script_abs_path}")

    if not os.path.isdir(output_abs_dir):
        # Create the output directory if it doesn't exist
        print(f"Output directory not found. Creating: {output_abs_dir}")
        os.makedirs(output_abs_dir)

    script_folder = os.path.dirname(script_abs_path)

    # 2. Dynamically build the command list
    command = [
        "python",
        script_abs_path,
        "--output_dir",
        output_abs_dir,
        "--task",
        task,
    ]

    # Add input_file only if it's not None
    if input_file:
        command.extend(["--input_file", input_abs_path])

    # Add general arguments
    if text_column:
        command.extend(["--text_column", text_column])
    if previous_output_files:
        command.extend(["--previous_output_files"] + previous_output_files)
    if username:
        command.extend(["--username", username])
    if save_to_user_folders is not None:
        command.extend(["--save_to_user_folders", str(save_to_user_folders)])
    if excel_sheets:
        command.append("--excel_sheets")
        command.extend(excel_sheets)
    if group_by:
        command.extend(["--group_by", group_by])

    # Add model configuration arguments
    if model_choice:
        command.extend(["--model_choice", model_choice])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if batch_size is not None:
        command.extend(["--batch_size", str(batch_size)])
    if max_tokens is not None:
        command.extend(["--max_tokens", str(max_tokens)])
    if api_url:
        command.extend(["--api_url", api_url])
    if inference_server_model:
        command.extend(["--inference_server_model", inference_server_model])

    # Add topic extraction arguments
    if context:
        command.extend(["--context", context])
    if candidate_topics:
        command.extend(["--candidate_topics", candidate_topics])
    if force_zero_shot:
        command.extend(["--force_zero_shot", force_zero_shot])
    if force_single_topic:
        command.extend(["--force_single_topic", force_single_topic])
    if produce_structured_summary:
        command.extend(["--produce_structured_summary", produce_structured_summary])
    if sentiment:
        command.extend(["--sentiment", sentiment])
    if additional_summary_instructions:
        command.extend(["--additional_summary_instructions", additional_summary_instructions])

    # Add validation arguments
    if additional_validation_issues:
        command.extend(["--additional_validation_issues", additional_validation_issues])
    if show_previous_table:
        command.extend(["--show_previous_table", show_previous_table])
    if output_debug_files:
        command.extend(["--output_debug_files", output_debug_files])
    if max_time_for_loop is not None:
        command.extend(["--max_time_for_loop", str(max_time_for_loop)])

    # Add deduplication arguments
    if method:
        command.extend(["--method", method])
    if similarity_threshold is not None:
        command.extend(["--similarity_threshold", str(similarity_threshold)])
    if merge_sentiment:
        command.extend(["--merge_sentiment", merge_sentiment])
    if merge_general_topics:
        command.extend(["--merge_general_topics", merge_general_topics])

    # Add summarisation arguments
    if summary_format:
        command.extend(["--summary_format", summary_format])
    if sample_reference_table:
        command.extend(["--sample_reference_table", sample_reference_table])
    if no_of_sampled_summaries is not None:
        command.extend(["--no_of_sampled_summaries", str(no_of_sampled_summaries)])
    if random_seed is not None:
        command.extend(["--random_seed", str(random_seed)])

    # Add output format arguments
    if create_xlsx_output is False:
        command.append("--no_xlsx_output")

    # Add logging arguments
    if save_logs_to_csv is not None:
        command.extend(["--save_logs_to_csv", str(save_logs_to_csv)])
    if save_logs_to_dynamodb is not None:
        command.extend(["--save_logs_to_dynamodb", str(save_logs_to_dynamodb)])
    if cost_code:
        command.extend(["--cost_code", cost_code])

    # Filter out None values before joining
    command_str = " ".join(str(arg) for arg in command if arg is not None)
    print(f"Executing command: {command_str}")

    # 3. Execute the command using subprocess
    try:
        # Use unbuffered output to avoid hanging
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        # Ensure inference server is enabled for testing
        env['RUN_INFERENCE_SERVER'] = '1'
        # Enable mock mode
        env['USE_MOCK_LLM'] = '1'
        env['TEST_MODE'] = '1'
        
        result = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout to avoid deadlocks
            text=True,
            cwd=script_folder,  # Important for relative paths within the script
            env=env,
            bufsize=0,  # Unbuffered
        )

        # Read output in real-time to avoid deadlocks
        start_time = time.time()
        
        # For Windows, we need a different approach
        if sys.platform == 'win32':
            # On Windows, use communicate with timeout
            try:
                stdout, stderr = result.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                result.kill()
                stdout, stderr = result.communicate()
                raise subprocess.TimeoutExpired(result.args, timeout)
        else:
            # On Unix, we can use select for real-time reading
            import select
            stdout_lines = []
            while result.poll() is None:
                ready, _, _ = select.select([result.stdout], [], [], 0.1)
                if ready:
                    line = result.stdout.readline()
                    if line:
                        print(line.rstrip(), flush=True)
                        stdout_lines.append(line)
                # Check timeout
                if time.time() - start_time > timeout:
                    result.kill()
                    raise subprocess.TimeoutExpired(result.args, timeout)
            
            # Read remaining output
            remaining = result.stdout.read()
            if remaining:
                print(remaining, end='', flush=True)
                stdout_lines.append(remaining)
            
            stdout = ''.join(stdout_lines)
            stderr = ''  # Combined with stdout

        print("--- SCRIPT STDOUT ---")
        if stdout:
            print(stdout)
        print("--- SCRIPT STDERR ---")
        if stderr:
            print(stderr)
        print("---------------------")

        # Analyze the output for errors and success indicators
        analysis = analyze_test_output(stdout, stderr)

        if analysis["has_errors"]:
            print("❌ Errors detected in output:")
            for i, error_type in enumerate(analysis["error_types"]):
                print(f"   {i+1}. {error_type}")
            if analysis["error_messages"]:
                print("   Error messages:")
                for msg in analysis["error_messages"][:3]:  # Show first 3 error messages
                    print(f"     - {msg}")
            return False
        elif result.returncode == 0:
            success_msg = "✅ Script executed successfully."
            if analysis["success_indicators"]:
                success_msg += f" (Success indicators: {', '.join(analysis['success_indicators'][:3])})"
            print(success_msg)
            return True
        else:
            print(f"❌ Command failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        result.kill()
        print(f"❌ Subprocess timed out after {timeout} seconds.")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False


def analyze_test_output(stdout: str, stderr: str) -> dict:
    """
    Analyze test output to provide detailed error information.

    Args:
        stdout (str): Standard output from the test
        stderr (str): Standard error from the test

    Returns:
        dict: Analysis results with error details
    """
    combined_output = (stdout or "") + (stderr or "")

    analysis = {
        "has_errors": False,
        "error_types": [],
        "error_messages": [],
        "success_indicators": [],
        "warning_indicators": [],
    }

    # Error patterns
    error_patterns = {
        "An error occurred": "General error message",
        "Error:": "Error prefix",
        "Exception:": "Exception occurred",
        "Traceback": "Python traceback",
        "Failed to": "Operation failure",
        "Cannot": "Operation not possible",
        "Unable to": "Operation not possible",
        "KeyError:": "Missing key/dictionary error",
        "AttributeError:": "Missing attribute error",
        "TypeError:": "Type mismatch error",
        "ValueError:": "Invalid value error",
        "FileNotFoundError:": "File not found",
        "ImportError:": "Import failure",
        "ModuleNotFoundError:": "Module not found",
    }

    # Success indicators
    success_patterns = [
        "Successfully",
        "Completed",
        "Finished",
        "Processed",
        "Complete",
        "Output files saved",
    ]

    # Warning indicators
    warning_patterns = ["Warning:", "WARNING:", "Deprecated", "DeprecationWarning"]

    # Check for errors
    for pattern, description in error_patterns.items():
        if pattern.lower() in combined_output.lower():
            analysis["has_errors"] = True
            analysis["error_types"].append(description)

            # Extract the actual error message
            lines = combined_output.split("\n")
            for line in lines:
                if pattern.lower() in line.lower():
                    analysis["error_messages"].append(line.strip())

    # Check for success indicators
    for pattern in success_patterns:
        if pattern.lower() in combined_output.lower():
            analysis["success_indicators"].append(pattern)

    # Check for warnings
    for pattern in warning_patterns:
        if pattern.lower() in combined_output.lower():
            analysis["warning_indicators"].append(pattern)

    return analysis


class TestCLITopicsExamples(unittest.TestCase):
    """Test suite for CLI topic extraction examples from the epilog."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests."""
        cls.script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "cli_topics.py"
        )
        cls.example_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "example_data"
        )
        cls.temp_output_dir = tempfile.mkdtemp(prefix="test_output_")

        # Verify script exists
        if not os.path.isfile(cls.script_path):
            raise FileNotFoundError(f"CLI script not found: {cls.script_path}")

        print(f"Test setup complete. Script: {cls.script_path}")
        print(f"Example data directory: {cls.example_data_dir}")
        print(f"Temp output directory: {cls.temp_output_dir}")
        print("Using function mocking instead of HTTP server")

        # Debug: Check if example data directory exists and list contents
        if os.path.exists(cls.example_data_dir):
            print("Example data directory exists. Contents:")
            for item in os.listdir(cls.example_data_dir):
                item_path = os.path.join(cls.example_data_dir, item)
                if os.path.isfile(item_path):
                    print(f"  File: {item} ({os.path.getsize(item_path)} bytes)")
                else:
                    print(f"  Directory: {item}")
        else:
            print(f"Example data directory does not exist: {cls.example_data_dir}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after running tests."""
        if os.path.exists(cls.temp_output_dir):
            shutil.rmtree(cls.temp_output_dir)
        print(f"Cleaned up temp directory: {cls.temp_output_dir}")

    def test_extract_topics_default_settings(self):
        """Test: Extract topics from a CSV file with default settings"""
        print("\n=== Testing topic extraction with default settings ===")
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic extraction with default settings should succeed")
        print("✅ Topic extraction with default settings passed")

    def test_extract_topics_custom_model_and_context(self):
        """Test: Extract topics with custom model and context"""
        print("\n=== Testing topic extraction with custom model and context ===")
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            context="Social Care case notes for young people",
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic extraction with custom model and context should succeed")
        print("✅ Topic extraction with custom model and context passed")

    def test_extract_topics_with_grouping(self):
        """Test: Extract topics with grouping"""
        print("\n=== Testing topic extraction with grouping ===")
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            group_by="Client",
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic extraction with grouping should succeed")
        print("✅ Topic extraction with grouping passed")

    def test_extract_topics_with_candidate_topics(self):
        """Test: Extract topics with candidate topics (zero-shot)"""
        print("\n=== Testing topic extraction with candidate topics ===")
        input_file = os.path.join(
            self.example_data_dir, "dummy_consultation_response.csv"
        )
        candidate_topics_file = os.path.join(
            self.example_data_dir, "dummy_consultation_response_themes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")
        if not os.path.isfile(candidate_topics_file):
            self.skipTest(f"Candidate topics file not found: {candidate_topics_file}")

        result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Response text",
            output_dir=self.temp_output_dir,
            candidate_topics=candidate_topics_file,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic extraction with candidate topics should succeed")
        print("✅ Topic extraction with candidate topics passed")

    def test_deduplicate_topics_fuzzy(self):
        """Test: Deduplicate topics using fuzzy matching"""
        print("\n=== Testing topic deduplication with fuzzy matching ===")
        
        # First, we need to create some output files by running extraction
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Run extraction first to create output files
        extract_result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        if not extract_result:
            self.skipTest("Extraction failed, cannot test deduplication")

        # Find the output files (they should be in temp_output_dir)
        # The file names follow a pattern like: {input_file_name}_col_{text_column}_reference_table.csv
        import glob
        reference_files = glob.glob(
            os.path.join(self.temp_output_dir, "*reference_table.csv")
        )
        unique_files = glob.glob(
            os.path.join(self.temp_output_dir, "*unique_topics.csv")
        )

        if not reference_files or not unique_files:
            self.skipTest("Could not find output files from extraction")

        result = run_cli_topics(
            script_path=self.script_path,
            task="deduplicate",
            previous_output_files=[reference_files[0], unique_files[0]],
            output_dir=self.temp_output_dir,
            method="fuzzy",
            similarity_threshold=90,
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic deduplication with fuzzy matching should succeed")
        print("✅ Topic deduplication with fuzzy matching passed")

    def test_deduplicate_topics_llm(self):
        """Test: Deduplicate topics using LLM"""
        print("\n=== Testing topic deduplication with LLM ===")
        
        # First, we need to create some output files by running extraction
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        # Run extraction first to create output files
        extract_result = run_cli_topics(
            script_path=self.script_path,
            task="extract",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        if not extract_result:
            self.skipTest("Extraction failed, cannot test deduplication")

        # Find the output files
        import glob
        reference_files = glob.glob(
            os.path.join(self.temp_output_dir, "*reference_table.csv")
        )
        unique_files = glob.glob(
            os.path.join(self.temp_output_dir, "*unique_topics.csv")
        )

        if not reference_files or not unique_files:
            self.skipTest("Could not find output files from extraction")

        result = run_cli_topics(
            script_path=self.script_path,
            task="deduplicate",
            previous_output_files=[reference_files[0], unique_files[0]],
            output_dir=self.temp_output_dir,
            method="llm",
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
        )

        self.assertTrue(result, "Topic deduplication with LLM should succeed")
        print("✅ Topic deduplication with LLM passed")

    def test_all_in_one_pipeline(self):
        """Test: Run complete pipeline (extract, deduplicate, summarise)"""
        print("\n=== Testing all-in-one pipeline ===")
        input_file = os.path.join(
            self.example_data_dir, "combined_case_notes.csv"
        )

        if not os.path.isfile(input_file):
            self.skipTest(f"Example file not found: {input_file}")

        result = run_cli_topics(
            script_path=self.script_path,
            task="all_in_one",
            input_file=input_file,
            text_column="Case Note",
            output_dir=self.temp_output_dir,
            model_choice="test-model",
            inference_server_model="test-model",
            api_url="http://localhost:8080",  # URL doesn't matter with function mocking
            create_xlsx_output=False,
            save_logs_to_csv=False,
            timeout=120,  # Shorter timeout for debugging
        )

        self.assertTrue(result, "All-in-one pipeline should succeed")
        print("✅ All-in-one pipeline passed")


def run_all_tests():
    """Run all test examples and report results."""
    print("=" * 80)
    print("LLM TOPIC MODELLER TEST SUITE")
    print("=" * 80)
    print("This test suite includes:")
    print("- CLI examples from the epilog")
    print("- GUI application tests")
    print("- Tests use a mock inference-server to avoid API costs")
    print("Tests will be skipped if required example files are not found.")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add CLI tests
    cli_suite = loader.loadTestsFromTestCase(TestCLITopicsExamples)
    suite.addTests(cli_suite)

    # Add GUI tests
    try:
        from test.test_gui_only import TestGUIAppOnly
        gui_suite = loader.loadTestsFromTestCase(TestGUIAppOnly)
        suite.addTests(gui_suite)
        print("GUI tests included in test suite.")
    except ImportError as e:
        print(f"Warning: Could not import GUI tests: {e}")
        print("Skipping GUI tests.")

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'✅ PASSED' if success else '❌ FAILED'}")
    print("=" * 80)

    return success


if __name__ == "__main__":
    # Run the test suite
    success = run_all_tests()
    exit(0 if success else 1)

