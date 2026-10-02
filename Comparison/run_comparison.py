import json
import os
import pandas as pd
import re

# 1. Load data
comparison_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(comparison_dir, 'key_3_details_updated_scores.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

project_dir = os.path.dirname(comparison_dir)
inputs_dir = os.path.join(project_dir, 'baselines', 'results')
df_pairs = pd.read_csv(os.path.join(inputs_dir, 'pairs_2b.csv'))
pairs_prompts = set(df_pairs['prompt'].tolist())

df_scores = pd.read_csv(os.path.join(inputs_dir, 'scores_2b.csv'))
csv_prompt_to_score = {}
for _, row in df_scores.iterrows():
    if row['prompt'] in pairs_prompts:
        csv_prompt_to_score[row['prompt']] = row['score']

# 2. Extract JSON prompts strictly from Layer 16
all_json_prompts = set()
top_keys = ['hierarchical features', 'avg activation features', 'Single-token driven']

for method in top_keys:
    if method in data:
        for dataset, contents in data[method].items():
            for l1_key, l2_dict in contents.items():
                # Only look inside Layer 16
                if isinstance(l2_dict, dict) and '16' in l2_dict:
                    generations = l2_dict['16']
                    for gen in generations:
                        if isinstance(gen, dict) and "user_prompt" in gen:
                            all_json_prompts.add(gen["user_prompt"])

# 3. Robust matching to handle newline/punctuation differences
def clean_string(s):
    return re.sub(r'[\W_]+', '', str(s)).lower()

json_to_csv_prompt = {}
for json_p in all_json_prompts:
    clean_json = clean_string(json_p)
    for csv_p in csv_prompt_to_score.keys():
        clean_csv = clean_string(csv_p)
        if clean_json in clean_csv or clean_csv in clean_json:
            json_to_csv_prompt[json_p] = csv_p
            break

# 4. Extract max steered scores exactly at Layer 16
prompt_data = {}

for method in top_keys:
    if method in data:
        for dataset, contents in data[method].items():
            for l1_key, l2_dict in contents.items():
                if isinstance(l2_dict, dict) and '16' in l2_dict:
                    generations = l2_dict['16']
                    for gen in generations:
                        if isinstance(gen, dict) and "user_prompt" in gen and "score" in gen:
                            prompt = gen["user_prompt"]
                            if prompt in json_to_csv_prompt:
                                score_str = gen["score"]
                                m = re.search(r'#Steered_Score:\s*(\d+)', str(score_str))
                                steered_score = int(m.group(1)) if m else 0
                                
                                # Format dataset names cleanly
                                clean_name = dataset.replace('dataset_', '').replace('.pkl', '').replace('_', ' ')
                                clean_name = clean_name.title()

                                if prompt not in prompt_data:
                                    prompt_data[prompt] = {
                                        "JSON_Prompt": prompt,
                                        "Dataset": clean_name,
                                        "Baseline_Score": csv_prompt_to_score[json_to_csv_prompt[prompt]],
                                        "Max_Steered_Score": steered_score
                                    }
                                else:
                                    if steered_score > prompt_data[prompt]["Max_Steered_Score"]:
                                        prompt_data[prompt]["Max_Steered_Score"] = steered_score

df_final = pd.DataFrame(list(prompt_data.values()))

# 5. Aggregate and count 4s and 5s
df_final['Steered_Score_4'] = (df_final['Max_Steered_Score'] == 4).astype(int)
df_final['Steered_Score_5'] = (df_final['Max_Steered_Score'] == 5).astype(int)
df_final['Baseline_Score_4'] = (df_final['Baseline_Score'] == 4).astype(int)
df_final['Baseline_Score_5'] = (df_final['Baseline_Score'] == 5).astype(int)

summary_df = df_final.groupby('Dataset').agg(
    Total_Prompts=('Dataset', 'count'),
    Baseline_Score_4=('Baseline_Score_4', 'sum'),
    Baseline_Score_5=('Baseline_Score_5', 'sum'),
    Steered_Score_4=('Steered_Score_4', 'sum'),
    Steered_Score_5=('Steered_Score_5', 'sum')
).reset_index()

# Add a TOTAL row
summary_df.loc['TOTAL'] = summary_df.sum(numeric_only=True)
summary_df.at['TOTAL', 'Dataset'] = 'OVERALL TOTAL'

# Export summary to CSV
results_dir = os.path.join(comparison_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'layer16_baseline_vs_steered.csv')
summary_df.to_csv(output_path, index=False)
print(f"\nExported to: {output_path}")