from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import gc
import os
import torch
import pickle
from scipy.cluster.hierarchy import to_tree
from sortedcontainers import SortedSet
os.makedirs("results_agglo_updated", exist_ok=True)

num_prompts = 100

with open(os.path.join("results_agglo_updated","filtered_feature_list_all_agglo_updated.pkl"), "rb") as f:
    filtered_feature_list_all = pickle.load(f) 

with open(os.path.join("results_agglo_updated", "select_tokens_agglo_updated.pkl"), "rb") as f:
    select_tokens = pickle.load(f)   

filtered_feature_list_all_to_list = {}
for data_files in os.listdir("dataset"):
     if data_files not in filtered_feature_list_all:
         continue
     if data_files not in filtered_feature_list_all_to_list:
         filtered_feature_list_all_to_list[data_files] = {}
     for prompt_no in range(0, num_prompts):
          if prompt_no not in filtered_feature_list_all[data_files]:
             continue
          if prompt_no not in filtered_feature_list_all_to_list[data_files]: 
             filtered_feature_list_all_to_list[data_files][prompt_no] = {}
          for lyr in range(0, 42):  
                if lyr not in filtered_feature_list_all[data_files][prompt_no]:
                   continue
                a = filtered_feature_list_all[data_files][prompt_no][lyr]
                
                filtered_feature_list_all_to_list[data_files][prompt_no][lyr] = (
                    a.detach().cpu().float().numpy()
                    if torch.is_tensor(a)
                    else np.asarray(a, dtype=np.float32)
                )

del filtered_feature_list_all
gc.collect()

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

     for prompt_no in range(0, num_prompts):
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

del filtered_feature_list_all_to_list
gc.collect()

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

     for prompt_no in range(0, num_prompts):
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

del linkage_all
del linkage_data_all
gc.collect()

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

     for prompt_no in range(0, num_prompts):
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