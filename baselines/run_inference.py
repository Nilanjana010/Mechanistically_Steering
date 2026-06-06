import os
import re
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_config import TARGET_MODEL_SIZE, get_model_id, get_results_prefix

# Step 1: Set paths and model identifiers.
base_results_dir = '/Mechanistically_Steering/baselines/results/'
output_csv = f'/Mechanistically_Steering/baselines/results/all_gemma_{TARGET_MODEL_SIZE}_inference_results.csv'

model_id = get_model_id(TARGET_MODEL_SIZE)
target_results_prefix = get_results_prefix(TARGET_MODEL_SIZE)
print(f"Loading model: {model_id} ...")

# Step 2: Load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Step 3: Compile text extraction patterns.
loss_pattern = re.compile(r"Final loss = ([\d\.]+)")
overlap_pattern = re.compile(r"Final overlap = ([\d\.]+)")
change_pattern = re.compile(r"Population attack overlap change = ([-?\d\.]+)")
x1_pattern = re.compile(r"Final x1:\s*(.*?)\nPopulation attack overlap change", re.DOTALL)

extracted_data = []

print(f"\nScanning directory tree: {base_results_dir}")

# Step 4: Read result files and generate model responses.
for root, dirs, files in os.walk(base_results_dir):
    rel_path = os.path.relpath(root, base_results_dir)
    top_level_dir = rel_path.split(os.sep, 1)[0] if rel_path != "." else ""

    if top_level_dir and not top_level_dir.startswith(target_results_prefix):
        dirs[:] = []
        continue

    for filename in files:
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(root, filename)

        experiment_label = rel_path if rel_path != "." else "root"

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        x1_match = x1_pattern.search(content)
        loss_match = loss_pattern.search(content)
        overlap_match = overlap_pattern.search(content)
        change_match = change_pattern.search(content)

        if not x1_match:
            print(f"[!] Found {filename}, but regex failed to extract 'Final x1:'")
            continue

        final_x1 = x1_match.group(1).replace('\n', '').strip()
        final_loss = loss_match.group(1) if loss_match else "N/A"
        final_overlap = overlap_match.group(1) if overlap_match else "N/A"
        overlap_change = change_match.group(1) if change_match else "N/A"

        inputs = tokenizer(final_x1, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        response_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[-1]:],
            skip_special_tokens=True
        )

        record = {
            "experiment_folder": experiment_label,
            "model_size": TARGET_MODEL_SIZE,
            "file": filename,
            "final_loss": final_loss,
            "final_overlap": final_overlap,
            "overlap_change": overlap_change,
            "adversarial_prompt": final_x1,
            "model_response": response_text.strip()
        }
        extracted_data.append(record)
        print(f"Processed {filename} from {experiment_label}")

    # Step 5: Write all extracted records to CSV.
if extracted_data:
    print(f"\nExporting {len(extracted_data)} total records to {output_csv}...")

    headers = ["experiment_folder", "model_size", "file", "final_loss", "final_overlap", "overlap_change", "adversarial_prompt", "model_response"]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(extracted_data)

    print("Export complete!")
else:
    print("No valid data found to export.")