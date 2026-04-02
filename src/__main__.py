
import argparse
import json
import sys
from pathlib import Path
from typing import List

from src.models import Function, Calling
from src.test import ConstrainedDecoder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with functions_definition, input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Function calling via constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=None,
        help="Path to the functions definition JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the function calling tests JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write the output JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the function calling pipeline.

    Steps:
    1. Parse CLI arguments for input/output paths.
    2. Resolve default paths relative to the project root.
    3. Validate that required JSON input files exist.
    4. Load and parse JSON data into Pydantic models.
    5. Run the constrained decoding engine.
    6. Write valid JSON output.
    """
    try:
        args = parse_args()

        base_dir = Path(__file__).resolve().parent.parent
        default_input = base_dir / "data" / "input"
        default_output = base_dir / "data" / "output"
        functions_path = Path(
            args.functions_definition
            if args.functions_definition
            else str(default_input / "functions_definition.json")
        )
        input_path = Path(
            args.input
            if args.input
            else str(
                default_input / "function_calling_tests.json"
            )
        )
        output_path = Path(
            args.output
            if args.output
            else str(
                default_output / "function_calling_results.json"
            )
        )
        if not functions_path.exists():
            print(
                "Error: functions definition file not found: "
                + str(functions_path)
            )
            sys.exit(1)
        if not input_path.exists():
            print(
                "Error: input file not found: "
                + str(input_path)
            )
            sys.exit(1)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(functions_path, "r", encoding="utf-8") as fh:
            try:
                raw_funcs: List[dict] = json.load(fh)
            except json.JSONDecodeError as err:
                print(
                    "Error: invalid JSON in "
                    + str(functions_path)
                    + ": " + str(err)
                )
                sys.exit(1)
        with open(input_path, "r", encoding="utf-8") as fh:
            try:
                raw_tests: List[dict] = json.load(fh)
            except json.JSONDecodeError as err:
                print(
                    "Error: invalid JSON in "
                    + str(input_path)
                    + ": " + str(err)
                )
                sys.exit(1)
        functions = [Function(**item) for item in raw_funcs]
        callables = [Calling(**item) for item in raw_tests]
        print(
            "Loaded "
            + str(len(functions))
            + " functions and "
            + str(len(callables))
            + " prompts."
        )
        decoder = ConstrainedDecoder()
        decoder.run(functions, callables, str(output_path))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print("Error: " + str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
