from scipy.cluster.hierarchy import linkage
import os
import torch
import pickle
import numpy as np
from scipy.cluster.hierarchy import to_tree            
os.makedirs("results_agglo_updated_linkage", exist_ok=True)

exported_caches_sae_input_folder = "exported_caches"

all_pts_sae_input = {}

for fname in os.listdir(exported_caches_sae_input_folder):
    print("fname = ", fname)
    for f_name in os.listdir(os.path.join(exported_caches_sae_input_folder, fname)):  
        path = os.path.join(os.path.join(exported_caches_sae_input_folder, fname), f_name)
        print("Loading:", f_name)
        if fname not in all_pts_sae_input:
            all_pts_sae_input[fname] = {}
        all_pts_sae_input[fname][f_name] = torch.load(path, map_location="cuda:0")        

with open(os.path.join("pos_lists_results", "cosine_similarity.pkl"), "rb") as f:
      loaded_cosine_sim = pickle.load(f)

token_ref = {}
for data_files in os.listdir("dataset"):
     if data_files not in loaded_cosine_sim:
          continue
     if data_files not in token_ref:
           token_ref[data_files] = {}
     for prompt_no, concepts in loaded_cosine_sim[data_files].items():
        if not concepts:
             continue
        tok_cos_list = []
        for concept_name, tok in concepts.items():
          if not tok:
               continue
          for token_no, cos in tok.items():
               tok_cos_list.append([token_no, cos])
        if tok_cos_list:       
            best_token = sorted(tok_cos_list, key=lambda x: x[1], reverse=True)
            print("best token is = ", best_token, data_files, prompt_no)
            best_token = best_token[0][0]  
            token_ref[data_files][prompt_no] = int(best_token)      

with open(os.path.join("results_agglo_updated_linkage", "token_ref_agglo_updated_linkage.pkl"), "wb") as f:
    pickle.dump(token_ref, f) 

feature_indices_list = {}
for data_files in os.listdir("dataset"):
     if data_files not in token_ref:
          continue
     if data_files not in feature_indices_list:
           feature_indices_list[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in token_ref[data_files]:
               continue
          if prompt_no not in feature_indices_list[data_files]:
               feature_indices_list[data_files][prompt_no] = {}
          for num_layer in range(0, 42):  
                print(f"Layer no = {num_layer}")
                if num_layer not in feature_indices_list[data_files][prompt_no]:
                     feature_indices_list[data_files][prompt_no][num_layer] = {}
                acts = torch.load(os.path.join(os.path.join("exported_caches", "cache_fol_" + data_files), "cache_tensor_" + data_files + "_prompt_" + str(prompt_no) + "_layer_" + str(num_layer) + ".pt"), map_location="cuda:0")
                
                m = token_ref[data_files][prompt_no]
                vals, inds = torch.topk(acts[0, m, :], 2)
                print(f"Token no = {m}, vals = {vals}, inds = {inds}")
                feature_indices_list[data_files][prompt_no][num_layer][m] = inds.tolist()

with open(os.path.join("results_agglo_updated_linkage", "feature_indices_list_agglo_updated_linkage.pkl"), "wb") as f:
    pickle.dump(feature_indices_list, f) 

filtered_feature_list_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in feature_indices_list:
          continue
     if data_files not in filtered_feature_list_all:
         filtered_feature_list_all[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in feature_indices_list[data_files]:
               continue
          if prompt_no not in filtered_feature_list_all[data_files]: 
             filtered_feature_list_all[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
                if lyr not in feature_indices_list[data_files][prompt_no]:
                     continue
                dir_3, dir_4 = "cache_fol_" + data_files, "cache_tensor_" + data_files + "_prompt_" + str(prompt_no) + "_layer_" + str(lyr) + ".pt"
                print("dir_3 == ", dir_3, "\n")
                print("dir_4 == ", dir_4)
                tnsr = all_pts_sae_input[dir_3][dir_4]
        
                new_data = tnsr.squeeze(0)
                print(f"new_data shape for new_data = ", new_data.shape)
                filtered_feature_list = []
                filtered_feature_list.append(new_data[token_ref[data_files][prompt_no]])
                print(f"Length of filtered_feature_list = ", len(filtered_feature_list), len(filtered_feature_list[0]))        
                print(f"filtered_feature_list = ", filtered_feature_list)
                new_data_1 = torch.stack(filtered_feature_list)
                print(f"new_data_1 shape before transpose = ", new_data_1.shape) 
                new_data_1 = new_data_1.transpose(0, 1)
                print(f"new_data_1 shape after transpose  = ", new_data_1.shape)

                X_new_data = new_data_1.cpu().numpy()  # Convert to NumPy array if it's a CUDA tensor

                n_bad = (~np.isfinite(X_new_data)).sum()
                if n_bad:
                    print(f"{data_files}/{prompt_no}/lyr{lyr}: cleaned {n_bad} non-finite values")
                X_new_data = np.nan_to_num(X_new_data, nan=0.0, posinf=0.0, neginf=0.0)

                filtered_feature_list_all[data_files][prompt_no][lyr] = X_new_data
                print("Dataset shape is ::", X_new_data.shape)

with open(os.path.join("results_agglo_updated_linkage", "filtered_feature_list_all_agglo_updated_linkage.pkl"), "wb") as f:
    pickle.dump(filtered_feature_list_all, f) 

list_of_linkage = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
         continue
     if data_files not in list_of_linkage:
         list_of_linkage[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in filtered_feature_list_all[data_files]:
             continue
          if prompt_no not in list_of_linkage[data_files]: 
             list_of_linkage[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
                if lyr not in filtered_feature_list_all[data_files][prompt_no]:
                   continue
                a = filtered_feature_list_all[data_files][prompt_no][lyr]
                list_of_linkage[data_files][prompt_no][lyr] = a.tolist()

with open(os.path.join("results_agglo_updated_linkage", "list_of_linkage_agglo_updated_linkage.pkl"), "wb") as f:
    pickle.dump(list_of_linkage, f) 

def ret_path(feature_indx, linkage_lst, pts):
    track_path = [feature_indx]
    d = []
    current_cluster = feature_indx
    for row_idx, (c1, c2, dist, size) in enumerate(linkage_lst):
        new_cluster_id = pts + row_idx
        if c1 == current_cluster or c2 == current_cluster:
            print("cluster :", current_cluster, " merged to form new cluster ", new_cluster_id, " with distance ", dist, " and size ", size, " c1 is ", c1, " c2 is ", c2)
            track_path.append(new_cluster_id)
            d.append((int(c1), int(c2), float(dist), int(size)))
            current_cluster = new_cluster_id
    return track_path, d

linkage_all = {}
linkage_data_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in list_of_linkage:
         continue
     if data_files not in linkage_all:
         linkage_all[data_files] = {}    
     if data_files not in linkage_data_all:
         linkage_data_all[data_files] = {}

     for prompt_no in range(0, 20):
          if prompt_no not in list_of_linkage[data_files]: 
             continue
          if prompt_no not in linkage_all[data_files]:
             linkage_all[data_files][prompt_no] = {}    
          if prompt_no not in linkage_data_all[data_files]:
             linkage_data_all[data_files][prompt_no] = {}

          for lyr in range(0, 42): 
             if lyr not in list_of_linkage[data_files][prompt_no]: 
                continue 
             linkage_data_all[data_files][prompt_no][lyr] = linkage(list_of_linkage[data_files][prompt_no][lyr], method='ward', metric='euclidean')

          for lyr in range(0, 42):
              for ft in feature_indices_list[data_files][prompt_no][lyr][token_ref[data_files][prompt_no]]:
                    if lyr not in linkage_all[data_files][prompt_no]:
                        linkage_all[data_files][prompt_no][lyr] = {}
                    ret_return, d_1 = ret_path(ft, linkage_data_all[data_files][prompt_no][lyr], 16384)
                    print(f'd_1 for {data_files}, {prompt_no}, {lyr}, and {ft} = ', d_1)
                    linkage_all[data_files][prompt_no][lyr][ft] = d_1 

with open(os.path.join("results_agglo_updated_linkage", f"linkage_all_linkage.pkl"), "wb") as f:
                pickle.dump(linkage_all, f)
with open(os.path.join("results_agglo_updated_linkage", f"linkage_data_all_linkage.pkl"), "wb") as f:
                pickle.dump(linkage_data_all, f)  

def rec(cluster_no, tree_inf):
    print("cluster_no =", cluster_no)
    tree, nodes = to_tree(tree_inf, rd=True)

    def get_leaves(node):
        """Recursively collect all leaf indices under a given node."""
        if node.is_leaf():
            return [node.id]
        else:
            return get_leaves(node.left) + get_leaves(node.right)
        
    cluster_node = nodes[cluster_no]
    print("cluster_node =", cluster_node)
    leaf_indices = get_leaves(cluster_node)
    print("leaf_indices =", leaf_indices)
    return leaf_indices

features_extract_linkage_all_layers = {}

for data_files in os.listdir("dataset"):
     if data_files not in feature_indices_list:
         continue
     if data_files not in features_extract_linkage_all_layers:
         features_extract_linkage_all_layers[data_files] = {}    

     for prompt_no in range(0, 20):
          if prompt_no not in feature_indices_list[data_files]: 
               continue  
          if prompt_no not in features_extract_linkage_all_layers[data_files]:
             features_extract_linkage_all_layers[data_files][prompt_no] = {}  

          for lyr in range(0, 42): 
             if lyr not in feature_indices_list[data_files][prompt_no]:
                  continue 
             if lyr not in features_extract_linkage_all_layers[data_files][prompt_no]:
                  features_extract_linkage_all_layers[data_files][prompt_no][lyr] = {}

             m = token_ref[data_files][prompt_no]
             visited = feature_indices_list[data_files][prompt_no][lyr][m]
     
             for ft in visited:
                print(f'in line 231 d_1 for {data_files}, {prompt_no}, {lyr}, and {ft}')
                d_ft = linkage_all[data_files][prompt_no][lyr][ft]
                print("d_ft = ", d_ft)
                features_extract_linkage_each = []
                for y in range(len(d_ft)):
            
                  print("len of features_extract_linkage_each before is ", len(features_extract_linkage_each))

                  if len(features_extract_linkage_each) < 25 and d_ft[y][3] < 50:

                    if d_ft[y][0] < 16384: 
                        features_extract_linkage_each.append([ft, lyr, d_ft[y][0]])
                    else:
                            print("len of features_extract_linkage_each after is ", len(features_extract_linkage_each))
                            lst_mn = rec(d_ft[y][0], linkage_data_all[data_files][prompt_no][lyr])
                            for w in lst_mn:
                                features_extract_linkage_each.append([ft, lyr, w])

                    if d_ft[y][1] < 16384:
                        features_extract_linkage_each.append([ft, lyr, d_ft[y][1]])
                    else:
                           print("len of features_extract_linkage_each before is ", len(features_extract_linkage_each))
                           lst_many = rec(d_ft[y][1], linkage_data_all[data_files][prompt_no][lyr])
                           for w in lst_many:
                               features_extract_linkage_each.append([ft, lyr, w])
                  else:
                      break             
                
                features_extract_linkage_all_layers[data_files][prompt_no][lyr][ft] = features_extract_linkage_each

with open(os.path.join("results_agglo_updated_linkage", f"features_extract_linkage_all_layers_linkage.pkl"), "wb") as f:
             pickle.dump(features_extract_linkage_all_layers, f) 

def remove_duplicates(list_of_lists):
    seen = set()
    result = []
    for sublist in list_of_lists:
        t = tuple(sublist)
        if t not in seen:
            seen.add(t)
            result.append(sublist)
    print("result =", result)        
    return result

features_extract_linkage_all_layers_to_non_duplicate = {}

for data_files in os.listdir("dataset"):
     if data_files not in features_extract_linkage_all_layers:
         continue
     if data_files not in features_extract_linkage_all_layers_to_non_duplicate:
         features_extract_linkage_all_layers_to_non_duplicate[data_files] = {}    

     for prompt_no in range(0, 20):
          if prompt_no not in features_extract_linkage_all_layers[data_files]: 
               continue 
          if prompt_no not in features_extract_linkage_all_layers_to_non_duplicate[data_files]:
             features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no] = {}    
      
          for lyr in range(0, 42):  
             if lyr not in features_extract_linkage_all_layers[data_files][prompt_no]:
                  continue
             if lyr not in features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no]:
                  features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr] = {}

             m = token_ref[data_files][prompt_no]
             visited = feature_indices_list[data_files][prompt_no][lyr][m]     

             for ft in visited:
                features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr][ft] = remove_duplicates(features_extract_linkage_all_layers[data_files][prompt_no][lyr][ft])

with open(os.path.join("results_agglo_updated_linkage", f"features_extract_linkage_all_layers_to_non_duplicate_linkage.pkl"), "wb") as f:
            pickle.dump(features_extract_linkage_all_layers_to_non_duplicate, f)
           