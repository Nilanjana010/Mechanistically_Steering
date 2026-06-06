import os
import subprocess
import torch
import shutil
from model_config import TARGET_MODEL_SIZE, get_model_type

# Step 1: Load target model settings.
target_model_type = get_model_type(TARGET_MODEL_SIZE)

# Step 2: Set run location and dataset list.
RUN_DIR = '/mech_steer/sae_robustness'
DATASET_DIR = '/mech_steer/baselines/datasets'
total_samples = 999

datasets_to_run = [
    "dataset_animal_abuse",
    "dataset_child_abuse",
    "dataset_controversial_topics,politics",
    "dataset_discrimination,stereotype,injustice",
    "dataset_drug_abuse,weapons,banned_substance",
    "dataset_financial_crime,property_crime,theft",
    "dataset_hate_speech,offensive_language",
    "dataset_misinformation_regarding_ethics,laws_and_safety",
    "dataset_non_violent_unethical_behavior",
    "dataset_privacy_violation",
    "dataset_self_harm",
    "dataset_sexually_explicit,adult_content",
    "dataset_terrorism,organized_crime",
    "dataset_violence,aiding_and_abetting,incitement"
]

# Step 3: Run attacks with fallback batch sizes.
batch_sizes_to_try = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
for dataset_name in datasets_to_run:
    print(f"\n=======================================================")
    print(f"🚀 STARTING DATASET: {dataset_name}")
    print(f"=======================================================")

    fatal_error = False

    for i in range(total_samples):
        if fatal_error:
            break

        print(f"\nProcessing sample_idx: {i}...")

        for b_size in batch_sizes_to_try:
            print(f"Trying batch_size: {b_size}...")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            data_file = os.path.join(DATASET_DIR, dataset_name)

            command = (
                f"python main.py "
                f"--mode suffix "
                f"--level population "
                f"--model_type {target_model_type} "
                f"--layer_num 16 "
                f"--data_file {data_file} "
                f"--sample_idx {i} "
                f"--suffix_len 3 "
                f"--batch_size {b_size} "
                f"--num_iters 20 "
                f"--m 300 "
                f"--k 170 "
                f"--log"
            )

            result = subprocess.run(
                command,
                shell=True,
                cwd=RUN_DIR,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode == 0:
                print(f"Success on sample_idx {i} with batch_size {b_size}.")
                break
            else:
                if "IndexError: single positional indexer is out-of-bounds" in result.stderr:
                    print(f"\nReached the end of '{dataset_name}' at sample_idx {i}. Moving to next dataset!")
                    fatal_error = True
                    break

                print(f"Failed with batch_size {b_size}.")

                if b_size == batch_sizes_to_try[-1]:
                    print(f"\nMemory error on sample_idx {i} even at batch size {b_size}. Skipping to next dataset.")
                    print(f"\n--- ERROR MESSAGE ---")
                    print(result.stderr)
                    print(f"---------------------\n")
                    fatal_error = True
                    break
                else:
                    print("Retrying with the next smaller batch size...")

    # Step 4: Copy dataset results to the final folder.
    print(f"\nFinished processing '{dataset_name}'. Copying to final destination...")
    source_folder = os.path.join(RUN_DIR, "results", f"{target_model_type}-{dataset_name}")

    if os.path.exists(source_folder):
        target_folder = os.path.join("/mech_steer/baselines/results", f"{target_model_type}-{dataset_name}")
        
        try:
            shutil.copytree(source_folder, target_folder, dirs_exist_ok=True)
            print(f"Successfully copied '{source_folder}' to {target_folder}!")
        except Exception as e:
            print(f"Error copying files: {e}")
    else:
        print(f"Warning: The expected output folder '{source_folder}' was not found. Nothing copied.")

# Step 5: Print final run status.
print("\nALL DATASETS COMPLETED!")