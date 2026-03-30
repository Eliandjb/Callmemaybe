import json
from pathlib import Path
from models import Function, Calling
from test import Startword


def main():
    """
    Main entry point for the function calling test suite.

    This function:
    1. Resolves input file paths for function definitions and test queries.
    2. Validates the existence of the required JSON input files.
    3. Loads and parses JSON data into Pydantic models (Function and Calling).
    4. Executes the LLM test suite via the Startword class to generate
       the function calling results.
    """
    try:
        BaseDire = Path(__file__).resolve().parent.parent
        input_dir = BaseDire / "data" / "input"
        functions_files = input_dir / "functions_definition.json"
        test_files = input_dir / "function_calling_tests.json"

        if not functions_files.exists() or not test_files.exists():
            print("Error : files inputs are empty")
            return

        with open(functions_files, "r") as fd1_def, \
             open(test_files, "r") as fd2_func:
            list_def = json.load(fd1_def)
            list_func = json.load(fd2_func)
            get_functions = [Function(**item) for item in list_def]
            get_callable = [Calling(**item) for item in list_func]

        tester = Startword()
        tester.test(get_functions, get_callable)
    except Exception as e:
        print(f"error: {e}")


if __name__ == "__main__":
    main()
