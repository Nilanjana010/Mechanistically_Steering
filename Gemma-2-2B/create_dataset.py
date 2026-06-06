import os
import pickle
import pandas as pd

splits = {
    '330k_train': 'round0/330k/train.jsonl.xz',
    '330k_test': 'round0/330k/test.jsonl.xz',
    '30k_train': 'round0/30k/train.jsonl.gz',
    '30k_test': 'round0/30k/test.jsonl.gz'
}
df = pd.read_json("hf://datasets/PKU-Alignment/BeaverTails/" + splits["330k_test"], lines=True)

categories = set()
for i in df['category'][0].keys():
    categories.add(i)
print("Categories = ", categories)    

output_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)  
for k in categories:
  ct, rows = 0, []
  for j in range(len(df['prompt'])):
    reset_default = 0
    for l, v in df.iloc[j]['category'].items():
      if v == True:
         reset_default += 1
    if reset_default == 1 and df['category'][j][k] == True:    
        if ct < 20:
          rows.append(df.iloc[j].to_dict())
          ct += 1
        elif ct >= 20:
          break      
  with open(os.path.join(output_dir, f"dataset_{k}.pkl"), "wb") as f:
        pickle.dump(rows, f)
  print("done")