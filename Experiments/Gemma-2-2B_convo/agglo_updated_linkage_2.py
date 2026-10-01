from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import torch
import pickle
from scipy.cluster.hierarchy import to_tree
from sortedcontainers import SortedSet
os.makedirs("results_agglo_updated", exist_ok=True)

num_prompts = 100
exported_caches_sae_input_folder = "exported_caches"

all_pts_sae_input = {}

for fname in os.listdir(exported_caches_sae_input_folder):
    print("fname = ", fname)
    if fname not in all_pts_sae_input:
            all_pts_sae_input[fname] = {}
    for f_name in os.listdir(os.path.join(exported_caches_sae_input_folder, fname)): 

        file_prompt = int(
            f_name.rsplit("_prompt_", 1)[1].split("_layer_", 1)[0]
        )
        print("file_prompt = ", file_prompt)
        if file_prompt >= num_prompts:
            continue

        path = os.path.join(os.path.join(exported_caches_sae_input_folder, fname), f_name)
        print("Loading:", f_name)
        all_pts_sae_input[fname][f_name] = torch.load(path, map_location="cpu")

with open(os.path.join("pos_lists_results", "cosine_similarity.pkl"), "rb") as f:
      loaded_cosine_sim = pickle.load(f)

select_tokens = {}      
for id1, data_File in loaded_cosine_sim.items():
    if id1 not in select_tokens:
        select_tokens[id1] = {}
    for id2, prompt_no in data_File.items():
        if int(id2) >= num_prompts:
           continue
        print("prompt_no = ", prompt_no)
        set_tokens = SortedSet()
        if not prompt_no:
            print("going to break inside prompt no....", prompt_no)
            continue
        if id2 not in select_tokens[id1]:
           select_tokens[id1][id2] = {} 
        for id3, concept in prompt_no.items():
            if not concept:
                print("going to break inside concept....", concept)
                continue
            for key in concept.keys():
                if key == 0:
                    continue
                print(f"key for {concept} = ", key)
                set_tokens.add(key)

        if set_tokens:
            print("set_tokens == ", set_tokens)
            dir_1 = "cache_fol_" + id1
            dir_2 = "cache_tensor_" + id1 + "_prompt_" + str(id2) +"_layer_20.pt" #16k
            acts = all_pts_sae_input[dir_1][dir_2]
            for m in set_tokens:   # loop over tokens
                    vals, inds = torch.topk(
                        acts[0, m, :], 1
                    )
                    print(f"Token no = {m}, vals = {vals}, inds = {inds}")
                       
                    select_tokens[id1][id2][m] = inds[0].item()  

with open(os.path.join("results_agglo_updated", "select_tokens_agglo_updated.pkl"), "wb") as f:
    pickle.dump(select_tokens, f) 

filtered_feature_list_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
         filtered_feature_list_all[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in select_tokens[data_files]:
               continue
          if not select_tokens[data_files][prompt_no]:
               continue
          if prompt_no not in filtered_feature_list_all[data_files]: 
             filtered_feature_list_all[data_files][prompt_no] = {}
          for lyr in range(0, 26):  
                dir_3, dir_4 = "cache_fol_" + data_files, "cache_tensor_" + data_files + "_prompt_" + str(prompt_no) + "_layer_" + str(lyr) + ".pt"
                print("dir_3 == ", dir_3, "\n")
                print("dir_4 == ", dir_4)
                tnsr = all_pts_sae_input[dir_3][dir_4]
        
                new_data = tnsr.squeeze(0)
                print(f"new_data shape for new_data = ", new_data.shape)
                filtered_feature_list = [] 
                for i in select_tokens[data_files][prompt_no].keys():
                        filtered_feature_list.append(new_data[i])
                print(f"Length of filtered_feature_list = ", len(filtered_feature_list))        
                
                new_data_1 = torch.stack(filtered_feature_list) 
                print(f"new_data_1 shape before transpose = ", new_data_1.shape) 
                new_data_1 = new_data_1.transpose(0, 1)
                print(f"new_data_1 shape after transpose  = ", new_data_1.shape)

                X_new_data = new_data_1.cpu().numpy()  # Convert to NumPy array if it's a CUDA tensor
                filtered_feature_list_all[data_files][prompt_no][lyr] = X_new_data
                print("Dataset shape is ::", X_new_data.shape)

with open(os.path.join("results_agglo_updated","filtered_feature_list_all_agglo_updated.pkl"), "wb") as f:
    pickle.dump(filtered_feature_list_all, f) 

labels_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
          continue
     if data_files not in labels_all:
           labels_all[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in filtered_feature_list_all[data_files]:
               continue
          if prompt_no not in labels_all[data_files]:
               labels_all[data_files][prompt_no] = {}
          for lyr in range(0, 26):  
                    hierarchical_cluster = AgglomerativeClustering(n_clusters=200, metric='euclidean', linkage='ward')
                    labels = hierarchical_cluster.fit_predict(filtered_feature_list_all[data_files][prompt_no][lyr]) 
                    labels_all[data_files][prompt_no][lyr] = labels

with open(os.path.join("results_agglo_updated", "labels_all_agglo_updated.pkl"), "wb") as f:
    pickle.dump(labels_all, f)  

layer_feature_label_dict = {}
updated_labels_clusters_final = {}

for data_files in os.listdir("dataset"):
     if data_files not in labels_all:
          continue
     if data_files not in layer_feature_label_dict:
           layer_feature_label_dict[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in labels_all[data_files]:
               continue
          if prompt_no not in layer_feature_label_dict[data_files]:
               layer_feature_label_dict[data_files][prompt_no] = {}
          for lyr in range(0, 26): 
               if lyr not in layer_feature_label_dict[data_files][prompt_no]:
                   layer_feature_label_dict[data_files][prompt_no][lyr] = {}
               lab_dict = labels_all[data_files][prompt_no][lyr]
               for label_index in range(lab_dict.shape[0]):
                   if label_index in select_tokens[data_files][prompt_no].values():
                       layer_feature_label_dict[data_files][prompt_no][lyr][label_index] = lab_dict[label_index]

for data_files in os.listdir("dataset"):
     if data_files not in layer_feature_label_dict:
          continue
     if data_files not in updated_labels_clusters_final:
           updated_labels_clusters_final[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in layer_feature_label_dict[data_files]:
               continue
          if prompt_no not in updated_labels_clusters_final[data_files]:
               updated_labels_clusters_final[data_files][prompt_no] = {}
          for lyr in range(0, 26):
               if lyr not in updated_labels_clusters_final[data_files][prompt_no]:
                   updated_labels_clusters_final[data_files][prompt_no][lyr] = {}
               lab_dict = labels_all[data_files][prompt_no][lyr]
               #print("lab_dict = ", lab_dict, lab_dict.shape[0])
               for label_index in range(lab_dict.shape[0]):
                   if lab_dict[label_index] in layer_feature_label_dict[data_files][prompt_no][lyr].values():
                       updated_labels_clusters_final[data_files][prompt_no][lyr][label_index] = lab_dict[label_index]               

with open(os.path.join("results_agglo_updated", "layer_feature_label_dict_agglo_updated.pkl"), "wb") as f:
    pickle.dump(layer_feature_label_dict, f)

with open(os.path.join("results_agglo_updated", "updated_labels_clusters_final_agglo_updated.pkl"), "wb") as f:
    pickle.dump(updated_labels_clusters_final, f)    

avg_feature_acts_layers  = {}
for data_files in os.listdir("dataset"):
     if data_files not in avg_feature_acts_layers:
         avg_feature_acts_layers[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in filtered_feature_list_all[data_files]: 
             continue
          if prompt_no not in avg_feature_acts_layers[data_files]:
              avg_feature_acts_layers[data_files][prompt_no] = {}
          for lyr in range(0, 26): 
              if lyr not in filtered_feature_list_all[data_files][prompt_no]: 
                  continue
              mn = filtered_feature_list_all[data_files][prompt_no][lyr]
              avg_feature_acts_layers[data_files][prompt_no][lyr] = np.mean(mn, axis = 1)

with open(os.path.join("results_agglo_updated", "avg_feature_acts_layers_agglo_updated.pkl"), "wb") as f:
    pickle.dump(avg_feature_acts_layers, f)
            