from pathlib import Path
import runpy


# Locate the Student ID-specific training script
SCRIPT_PATH = Path(__file__).resolve().parent / "train_23L-2554.py"


if __name__ == "__main__":
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")