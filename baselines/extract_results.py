import csv
import os
from model_config import TARGET_MODEL_SIZE

# Step 1: Set source and target CSV files.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'baselines', 'results')
INPUT_CSV  = os.path.join(RESULTS_DIR, f'all_gemma_{TARGET_MODEL_SIZE}_inference_results.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, f'pairs_{TARGET_MODEL_SIZE}.csv')


def extract_pairs(input_csv: str = INPUT_CSV, output_csv: str = OUTPUT_CSV):
    # Step 2: Read and collect valid prompt/response pairs.
    pairs = []

    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt   = row.get('adversarial_prompt', '').strip()
            response = row.get('model_response', '').strip()
            if prompt and response:
                pairs.append({'prompt': prompt, 'response': response})

    # Step 3: Write cleaned pairs to output CSV.
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['prompt', 'response'])
        writer.writeheader()
        writer.writerows(pairs)

    print(f"Extracted {len(pairs)} pairs → {output_csv}")


# Step 4: Run as script entrypoint.
if __name__ == "__main__":
    extract_pairs()