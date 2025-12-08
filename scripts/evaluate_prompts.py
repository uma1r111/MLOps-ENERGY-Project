import sys
from pathlib import Path

EVAL_DATASET = [
    {"query": "energy", "expected": True},
    {"query": "heating", "expected": True},
    {"query": "solar panels", "expected": True},
    {"query": "household energy use", "expected": True},
    {"query": "efficiency", "expected": True},
]


def evaluate_prompts():
    prompt_file = Path("experiments/prompts.py")

    # Convert Path to string and check if file exists
    if not prompt_file.exists():
        print("No prompts.py found, skipping evaluation")
        return 0

    # Read the content of the file
    with open(
        str(prompt_file), "r", encoding="utf-8"
    ) as f:  # Convert Path to string here
        content = f.read()

    passed = 0
    for test_case in EVAL_DATASET:
        if test_case["query"].lower() in content.lower():
            passed += 1

    score = passed / len(EVAL_DATASET)
    print(f"Prompt Evaluation Score: {score:.2%}")

    if score < 0.5:
        print(f"❌ Score {score:.2%} below threshold (50%)")
        return 1

    print(f"✓ Score {score:.2%} meets threshold")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate_prompts())
