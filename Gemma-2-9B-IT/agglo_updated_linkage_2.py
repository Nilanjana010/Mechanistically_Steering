from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import os
import torch
import pickle
from scipy.cluster.hierarchy import to_tree
from sortedcontainers import SortedSet
os.makedirs("results_agglo_updated", exist_ok=True)

exported_caches_sae_input_folder = "exported_caches"

all_pts_sae_input = {}

for fname in os.listdir(exported_caches_sae_input_folder):
    print("fname = ", fname)
    if fname not in all_pts_sae_input:
            all_pts_sae_input[fname] = {}
    for f_name in os.listdir(os.path.join(exported_caches_sae_input_folder, fname)):  
        path = os.path.join(os.path.join(exported_caches_sae_input_folder, fname), f_name)
        print("Loading:", f_name)
        all_pts_sae_input[fname][f_name] = torch.load(path, map_location="cuda:0")        

with open(os.path.join("pos_lists_results", "cosine_similarity.pkl"), "rb") as f:
      loaded_cosine_sim = pickle.load(f)

select_tokens = {}      
for id1, data_File in loaded_cosine_sim.items():
    if id1 not in select_tokens:
        select_tokens[id1] = {}
    for id2, prompt_no in data_File.items():
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
                    for val, ind in zip(vals, inds):
                        print(f"Feature is {ind} fired at {val:.2f}")
                         
                    select_tokens[id1][id2][m] = inds[0].item()  

with open(os.path.join("results_agglo_updated", "select_tokens_agglo_updated.pkl"), "wb") as f:
    pickle.dump(select_tokens, f) 

filtered_feature_list_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
         filtered_feature_list_all[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in select_tokens[data_files]:
               continue
          if not select_tokens[data_files][prompt_no]:
               continue
          if prompt_no not in filtered_feature_list_all[data_files]: 
             filtered_feature_list_all[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
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
                print(f"Shape of each feature in filtered_feature_list = ", filtered_feature_list)
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

with open(os.path.join("results_agglo_updated","filtered_feature_list_all_agglo_updated.pkl"), "wb") as f:
    pickle.dump(filtered_feature_list_all, f) 

labels_all = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
          continue
     if data_files not in labels_all:
           labels_all[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in filtered_feature_list_all[data_files]:
               continue
          if prompt_no not in labels_all[data_files]:
               labels_all[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
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
     for prompt_no in range(0, 20):
          if prompt_no not in labels_all[data_files]:
               continue
          if prompt_no not in layer_feature_label_dict[data_files]:
               layer_feature_label_dict[data_files][prompt_no] = {}
          for lyr in range(0, 42): 
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
     for prompt_no in range(0, 20):
          if prompt_no not in layer_feature_label_dict[data_files]:
               continue
          if prompt_no not in updated_labels_clusters_final[data_files]:
               updated_labels_clusters_final[data_files][prompt_no] = {}
          for lyr in range(0, 42):
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
     for prompt_no in range(0, 20):
          if prompt_no not in filtered_feature_list_all[data_files]: 
             continue
          if prompt_no not in avg_feature_acts_layers[data_files]:
              avg_feature_acts_layers[data_files][prompt_no] = {}
          for lyr in range(0, 42): 
              if lyr not in filtered_feature_list_all[data_files][prompt_no]: 
                  continue
              mn = filtered_feature_list_all[data_files][prompt_no][lyr]
              avg_feature_acts_layers[data_files][prompt_no][lyr] = np.mean(mn, axis = 1)

with open(os.path.join("results_agglo_updated", "avg_feature_acts_layers_agglo_updated.pkl"), "wb") as f:
    pickle.dump(avg_feature_acts_layers, f) 

filtered_feature_list_all_to_list = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
         continue
     if data_files not in filtered_feature_list_all_to_list:
         filtered_feature_list_all_to_list[data_files] = {}
     for prompt_no in range(0, 20):
          if prompt_no not in filtered_feature_list_all[data_files]:
             continue
          if prompt_no not in filtered_feature_list_all_to_list[data_files]: 
             filtered_feature_list_all_to_list[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
                if lyr not in filtered_feature_list_all[data_files][prompt_no]:
                   continue
                a = filtered_feature_list_all[data_files][prompt_no][lyr]
                print("a = ", a)
                filtered_feature_list_all_to_list[data_files][prompt_no][lyr] = a.tolist()

with open(os.path.join("results_agglo_updated", "filtered_feature_list_all_to_list_agglo_updated.pkl"), "wb") as f:
    pickle.dump(filtered_feature_list_all_to_list, f) 

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
     if data_files not in filtered_feature_list_all_to_list:
         continue
     if data_files not in linkage_all:
         linkage_all[data_files] = {}    
     if data_files not in linkage_data_all:
         linkage_data_all[data_files] = {}

     for prompt_no in range(0, 20):
          if prompt_no not in filtered_feature_list_all_to_list[data_files]: 
             continue
          if prompt_no not in linkage_all[data_files]:
             linkage_all[data_files][prompt_no] = {}    
          if prompt_no not in linkage_data_all[data_files]:
             linkage_data_all[data_files][prompt_no] = {}

          for lyr in range(0, 42):  
             linkage_data_all[data_files][prompt_no][lyr] = linkage(filtered_feature_list_all_to_list[data_files][prompt_no][lyr], method='ward', metric='euclidean')

          visited = set()
          for ft in select_tokens[data_files][prompt_no].values():
            if ft not in visited:
              visited.add(ft)
            else:
              continue  
            
            for lyr in range(0, 42):
                    if lyr not in linkage_all[data_files][prompt_no]:
                        linkage_all[data_files][prompt_no][lyr] = {}
                    ret_return, d_1 = ret_path(ft, linkage_data_all[data_files][prompt_no][lyr], 16384)
                    linkage_all[data_files][prompt_no][lyr][ft] = d_1 

with open(os.path.join("results_agglo_updated", f"linkage_all.pkl"), "wb") as f:
                pickle.dump(linkage_all, f)
with open(os.path.join("results_agglo_updated", f"linkage_data_all.pkl"), "wb") as f:
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
     if data_files not in select_tokens:
         continue
     if data_files not in features_extract_linkage_all_layers:
         features_extract_linkage_all_layers[data_files] = {}    

     for prompt_no in range(0, 20):
          if prompt_no not in select_tokens[data_files]: 
               continue
          if not select_tokens[data_files][prompt_no]:
               continue  
          if prompt_no not in features_extract_linkage_all_layers[data_files]:
             features_extract_linkage_all_layers[data_files][prompt_no] = {}    

          visited = set()
          for ft in select_tokens[data_files][prompt_no].values():
            if ft not in visited:
              visited.add(ft)
            else:
              continue 

          for lyr in range(0, 42):  
             if lyr not in features_extract_linkage_all_layers[data_files][prompt_no]:
                  features_extract_linkage_all_layers[data_files][prompt_no][lyr] = {}
             for ft in visited:
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

with open(os.path.join("results_agglo_updated", f"features_extract_linkage_all_layers.pkl"), "wb") as f:
             pickle.dump(features_extract_linkage_all_layers, f)

def remove_duplicates(list_of_lists):
    seen = set()
    result = []
    for sublist in list_of_lists:
        t = tuple(sublist)
        if t not in seen:
            seen.add(t)
            result.append(sublist)
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
          
          visited = set()
          for ft in select_tokens[data_files][prompt_no].values():
            if ft not in visited:
              visited.add(ft)
            else:
              continue 
      
          for lyr in range(0, 42):  
             if lyr not in features_extract_linkage_all_layers[data_files][prompt_no]:
                  continue
             if lyr not in features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no]:
                  features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr] = {}
             for ft in visited:
                features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr][ft] = remove_duplicates(features_extract_linkage_all_layers[data_files][prompt_no][lyr][ft])

with open(os.path.join("results_agglo_updated", f"features_extract_linkage_all_layers_to_non_duplicate.pkl"), "wb") as f:
            pickle.dump(features_extract_linkage_all_layers_to_non_duplicate, f)
            