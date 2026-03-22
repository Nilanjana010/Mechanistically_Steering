import os
import torch
import pickle
from huggingface_hub import login
login(token="****")
import torch.nn.functional as F

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

print("----------------------------------------------Starts here----------------------------------------------")

pkl_dir = "concept_vector_dictionaries"

all_pkls = {}

for fname in os.listdir(pkl_dir):
   for f_name in os.listdir(os.path.join(pkl_dir, fname)):
        print(f"Loading {fname}...{f_name}")
        new_path = os.path.join(os.path.join(pkl_dir, fname), f_name)
        if fname not in all_pkls:
            all_pkls[fname] = {}
        all_pkls[fname][f_name] = torch.load(new_path)

exported_caches_sae_input_folder = "exported_caches_20"

all_pts_sae_input = {}

for fname_2 in os.listdir(exported_caches_sae_input_folder):
        path = os.path.join(exported_caches_sae_input_folder, fname_2)
        print("Loading:", fname_2)
        all_pts_sae_input[fname_2] = torch.load(path)

folder_pos = "pos_lists_results"
os.makedirs(folder_pos, exist_ok=True)

cosine_similarity = {}

for data_files in os.listdir("dataset"):
     if data_files not in cosine_similarity:
       cosine_similarity[data_files] = {}
     for prompt_no in range(0, 20):
              dir_1 = "cache_tensor_20_" + data_files + "_prompt_" + str(prompt_no) +".pt"
              file_path_check = os.path.join(exported_caches_sae_input_folder, dir_1)
              if not os.path.exists(file_path_check):
                  continue
              if prompt_no not in cosine_similarity[data_files]:
                cosine_similarity[data_files][prompt_no] = {}
              print("dir_1 = ", dir_1, all_pts_sae_input[dir_1], all_pts_sae_input[dir_1].shape)
              open_dir = "concept_vector_concept_dictionary_" + data_files + "_dictionaries"
              for exp_1 in os.listdir(os.path.join("concept_vector_dictionaries", open_dir)): 
                   if "prompt_" + str(prompt_no) + "." in exp_1:
                        if exp_1 not in cosine_similarity[data_files][prompt_no]:
                          cosine_similarity[data_files][prompt_no][exp_1] = {}
                        print("exp_1 is = ", open_dir, exp_1, prompt_no, data_files)
                        print("exp_1 = ", all_pkls[open_dir][exp_1], all_pkls[open_dir][exp_1].shape)
            
                        x_cache = all_pts_sae_input[dir_1] 
                        print(f"Shape of x_cache is = ", x_cache.shape)                 
                        for i in range(x_cache.shape[1]):   # loop over tokens
                            vec = x_cache[0, i, :]    # shape [2304]
                            print("vec = ", i, vec.shape)
                            cos_sim = F.cosine_similarity(all_pkls[open_dir][exp_1], vec, dim=0)  # dim=0 since they’re 1D

                            if cos_sim > 0:
                               cosine_similarity[data_files][prompt_no][exp_1][i] = cos_sim.item()
                            print(f"cos_sim for {i} = ", cos_sim)

with open(os.path.join(folder_pos, "cosine_similarity.pkl"), "wb") as f:
    pickle.dump(cosine_similarity, f)                            
                        
print("----------------------------------------------Ends here----------------------------------------------")