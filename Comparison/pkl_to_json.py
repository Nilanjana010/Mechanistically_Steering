"""Convert key_3_details_updated_scores from pickle to JSON."""

import argparse
import json
import pickle
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PICKLE_PATH = SCRIPT_DIR / "key_3_details_updated_scores"
JSON_PATH = SCRIPT_DIR / "key_3_details_updated_scores.json"


def convert_pickle_to_json() -> Path:
    """Convert the project's trusted pickle file, replacing its JSON file."""
    with PICKLE_PATH.open("rb") as pickle_file:
        data = pickle.load(pickle_file)

    try:
        json_data = json.dumps(data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pickle contents cannot be represented as JSON: {exc}") from exc

    JSON_PATH.write_text(json_data + "\n", encoding="utf-8")
    return JSON_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert key_3_details_updated_scores to "
            "key_3_details_updated_scores.json. Only run this with a trusted "
            "pickle file, because loading pickle can execute code."
        )
    )
    parser.parse_args()

    try:
        output_path = convert_pickle_to_json()
    except (OSError, EOFError, pickle.UnpicklingError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Converted {PICKLE_PATH.name} to {output_path.name}")


if __name__ == "__main__":
    main()
