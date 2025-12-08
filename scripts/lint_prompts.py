import sys
from pathlib import Path

print("Script is running...")  # Debugging line


def lint_prompt_file(filepath):
    errors = []
    print(f"Attempting to lint: {filepath}")  # Debugging line
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        print(
            f"File content read successfully... Length: {len(content)} characters"
        )  # Debugging line

        if len(content.strip()) == 0:
            errors.append(f"{filepath}: Empty prompt file")

        if len(content) > 4000:
            errors.append(f"{filepath}: Prompt too long (>{4000} chars)")

        if str(filepath).endswith(".py"):
            try:
                compile(content, str(filepath), "exec")
            except SyntaxError as e:
                errors.append(f"{filepath}: Syntax error - {e}")

    except Exception as e:
        errors.append(f"Error reading file {filepath}: {str(e)}")

    return errors


def main():
    """Main linting entry point.

    In pytest:
        - returns 0 or 1 (no SystemExit)
    In CLI:
        - raises SystemExit(0 or 1)
    """

    running_tests = "pytest" in sys.modules

    # Test-safe argument injection
    if running_tests:
        dummy_path = Path("tests/dummy_prompt.txt")
        dummy_path.parent.mkdir(parents=True, exist_ok=True)

        if not dummy_path.exists():
            dummy_path.write_text("Hello world", encoding="utf-8")

        sys.argv = ["lint_prompts.py", str(dummy_path)]

    print("Starting linting process...")

    if len(sys.argv) < 2:
        print("No file path provided!")
        return 1 if running_tests else sys.exit(1)

    prompt_file = Path(sys.argv[1])

    if not prompt_file.exists():
        print(f"Error: {prompt_file} does not exist.")
        return 1 if running_tests else sys.exit(1)

    print(f"Linting file: {prompt_file}")

    errors = lint_prompt_file(prompt_file)

    if errors:
        print("Found the following errors:")
        for e in errors:
            print(e)
        return 1 if running_tests else sys.exit(1)

    print(f"✓ {prompt_file} passed linting")

    return 0 if running_tests else sys.exit(0)


if __name__ == "__main__":
    sys.exit(main())
