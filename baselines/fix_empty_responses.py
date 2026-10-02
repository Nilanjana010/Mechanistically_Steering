import os
import csv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_config import TARGET_MODEL_SIZE, get_model_id

# Step 1: Set file path and model id.
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
master_csv_path = os.path.join(
    project_dir,
    'baselines',
    'results',
    f'all_gemma_{TARGET_MODEL_SIZE}_inference_results.csv'
)
model_id = get_model_id(TARGET_MODEL_SIZE)

# Step 2: Load model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 
)

print(f"\nScanning master CSV for empty responses: {master_csv_path}")

rows = []
needs_saving = False
total_fixed = 0

# Step 3: Load all rows from the master CSV.
try:
    with open(master_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            rows.append(row)
except FileNotFoundError:
    print(f"Error: Could not find the master CSV at {master_csv_path}")
    exit(1)

# Step 4: Regenerate missing or empty responses.
for i, row in enumerate(rows):
    response = row.get('model_response', '').strip()
    
    if not response or response == "[MODEL_PRODUCED_EMPTY_OUTPUT]":
        prompt = row.get('adversarial_prompt', '').strip()
        file_name = row.get('file', 'Unknown')
        experiment = row.get('experiment_folder', 'Unknown')
        
        print(f"Fixing inference {i+1}/{len(rows)} (Exp: {experiment}, File: {file_name})...")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150, 
                do_sample=False     
            )
        
        response_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
        
        if not response_text:
            print("    -> Model returned an empty string (likely safety filter).")
            row['model_response'] = "[MODEL_PRODUCED_EMPTY_OUTPUT]"
        else:
            print("    -> Successfully generated text.")
            row['model_response'] = response_text
            
        needs_saving = True
        total_fixed += 1

    # Step 5: Save updates only when changes were made.
if needs_saving:
    print(f"\nSaving {total_fixed} fixes back to master CSV...")
    with open(master_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print("Master CSV updated successfully.")
else:
    print("\nNo empty responses found. Master CSV is already clean!")

# Step 6: Print final summary.
print(f"\nDone! Processed {len(rows)} total rows and patched {total_fixed} responses.")