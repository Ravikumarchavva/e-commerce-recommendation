from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)