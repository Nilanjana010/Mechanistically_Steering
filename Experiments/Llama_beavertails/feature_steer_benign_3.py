import os
import json
import time
import pickle
import requests
import numpy as np

url = "https://www.neuronpedia.org/api/steer"

evaluations = 0
key_details = {}

def remove_prompt(text, user_prompt):
    for prefix in ("<|begin_of_text|>" + user_prompt, user_prompt):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

def safety_check(features_to_steer, layer, triggered_feature, user_prompt, feature_type, data_file_name, prompt_number):
        global evaluations
        evaluations += 1

        MODEL_ID = "llama3.1-8b"
        FEATURE = []
        for f in features_to_steer:
            print(f"Feature to steer: {f} in layer {layer} for triggered feature {triggered_feature} of type {feature_type}")
            l = str(layer) + "-llamascope-res-32k"
            print(f"l is :: {l}")
            FEATURE.append({"modelId": MODEL_ID, "layer": l, "index": f, "strength": 5})
        print("FEATURE is :: ", FEATURE)

        # other settings
        TEMPERATURE = 0.2
        N_TOKENS = 256
        FREQ_PENALTY = 1.0
        SEED = 16
        STRENGTH_MULTIPLIER = 4

        # make the request
        data = {
            "prompt": user_prompt,
            "modelId": MODEL_ID,
            "features": FEATURE,
            "temperature": TEMPERATURE,
            "n_tokens": N_TOKENS,
            "freq_penalty": FREQ_PENALTY,
            "seed": SEED,
            "strength_multiplier": STRENGTH_MULTIPLIER,
        }
        headers = {"Content-Type": "application/json", "x-api-key": "****"}

        # send request
        print("data is :: ", data)
        response = requests.post(url, json=data, headers=headers)
        
        json_response = response.json()
        print("json_response is :: ", json_response)

        formatted_response = json.dumps(json_response, indent=4)
        print(f"formatted_response for the feature to steer {features_to_steer} in layer {layer} for triggered feature {triggered_feature} of type {feature_type} :", formatted_response)

        if 'STEERED' in json_response:
            default_response = json_response['DEFAULT']
            print("steered response before formatting = ", formatted_response)            
            formatted_response = remove_prompt(
                json_response['STEERED'], user_prompt
            )
            print("steered response after formatting = ", formatted_response)

            
            print("default response before formatting = ", default_response)            
            default_response = remove_prompt(
                json_response['DEFAULT'], user_prompt
            )
            print("default response after formatting = ", default_response)

            if feature_type not in key_details:
                 key_details[feature_type] = {}
            if data_file_name not in key_details[feature_type]:
                key_details[feature_type][data_file_name] = {}
            if prompt_number not in key_details[feature_type][data_file_name]:
                key_details[feature_type][data_file_name][prompt_number] = {}
            if layer not in key_details[feature_type][data_file_name][prompt_number]:
                key_details[feature_type][data_file_name][prompt_number][layer] = []
           
            key_details[feature_type][data_file_name][prompt_number][layer].append({'features_to_steer': features_to_steer, 'triggered_feature': triggered_feature, 'user_prompt': user_prompt, 'STEERED': formatted_response, 'DEFAULT': default_response})
            
            if json_response['limit'] == "1":
                 time.sleep(3600)
        else:
           print("Key 'STEERED' not found in response. Full response:", formatted_response)
        #    if json_response['limit'] == "1":
        #          time.sleep(3600)

top_f = 3
feature_set = set()

with open("benign_prompts.pkl", "rb") as f:
     benign = pickle.load(f)

with open(os.path.join("results_agglo_updated", "select_tokens_agglo_updated.pkl"), "rb") as f:
    select_tokens = pickle.load(f) 

with open(os.path.join("results_agglo_updated", "features_extract_linkage_all_layers_to_non_duplicate.pkl"), "rb") as f:
     features_extract_linkage_all_layers_to_non_duplicate = pickle.load(f)

     print("features_extract_linkage_all_layers_to_non_duplicate = ", features_extract_linkage_all_layers_to_non_duplicate)
     for data_files in os.listdir("dataset"):
        if data_files not in features_extract_linkage_all_layers_to_non_duplicate:
            continue

        for prompt_no in range(0, 20):
            if prompt_no not in features_extract_linkage_all_layers_to_non_duplicate[data_files]: 
                continue                 
            
            user_prompt = benign[data_files][prompt_no]['new'][11:]

            visited = set()
            for ft in select_tokens[data_files][prompt_no].values():
                if ft not in visited:
                    visited.add(ft)
                else:
                    continue 
        
            for lyr in range(0, 32):  

                if lyr not in features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no]:
                    continue
                for feature in visited:
                    if feature not in features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr]:
                        continue
                    if features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr]:
                        c = 1
                        f = []
                        en_ft = features_extract_linkage_all_layers_to_non_duplicate[data_files][prompt_no][lyr][feature]
                        for en in range(len(en_ft)):
                            if c <= top_f:
                                s = en_ft[en]
                                f.append(s[2])
                                c += 1
                        if f:        
                            print(f"features for layer {lyr} and {feature} are :: ", f)
                            safety_check(f, lyr, feature, user_prompt, "hierarchical features", data_files, prompt_no)        

print("Next is for the avg. feature activation method")
print("Evaluation 1 = ", evaluations)

with open("key_details_benign_1.pkl", "wb") as f:
    pickle.dump(key_details, f)

with open(os.path.join("results_agglo_updated", "avg_feature_acts_layers_agglo_updated.pkl"), "rb") as f:
    avg_feature_acts_layers = pickle.load(f) 
with open(os.path.join("results_agglo_updated", "updated_labels_clusters_final_agglo_updated.pkl"), "rb") as f:
    updated_labels_clusters_final = pickle.load(f) 

for data_files in os.listdir("dataset"):
     if data_files not in avg_feature_acts_layers:
         continue

     for prompt_no in range(0, 20):
          if prompt_no not in avg_feature_acts_layers[data_files]: 
             continue
          
          user_prompt = benign[data_files][prompt_no]['new'][11:]

          visited = set()
          for ft in select_tokens[data_files][prompt_no].values():
                if ft not in visited:
                    visited.add(ft)
                else:
                    continue

          for lyr in range(0, 32): 

              if lyr not in avg_feature_acts_layers[data_files][prompt_no]: 
                  continue
              sorted_avg_lst_indices = np.argsort(avg_feature_acts_layers[data_files][prompt_no][lyr])[::-1]
              for feature in visited:
                    c = 1
                    f = []
                    ref_cluster = updated_labels_clusters_final[data_files][prompt_no][lyr][feature]     
                    for en_num in sorted_avg_lst_indices:
                        if c > top_f:
                            break
                        if en_num not in updated_labels_clusters_final[data_files][prompt_no][lyr]:
                            continue
                        if c <= top_f and updated_labels_clusters_final[data_files][prompt_no][lyr][en_num] == ref_cluster:
                            f.append(int(en_num))
                            c += 1
                    if f:        
                        print(f"avg features for layer {lyr} and {feature} are :: ", f)
                        safety_check(f, lyr, feature, user_prompt, "avg activation features", data_files, prompt_no)

print("Evaluation 2 = ", evaluations)            

with open("key_details_benign_2.pkl", "wb") as f:
    pickle.dump(key_details, f)

with open(os.path.join("results_agglo_updated_linkage", f"features_extract_linkage_all_layers_to_non_duplicate_linkage.pkl"), "rb") as f:
            features_extract_linkage_all_layers_to_non_duplicate_2 = pickle.load(f)
with open(os.path.join("results_agglo_updated_linkage", "token_ref_agglo_updated_linkage.pkl"), "rb") as f:
            token_ref = pickle.load(f) 
with open(os.path.join("results_agglo_updated_linkage", "feature_indices_list_agglo_updated_linkage.pkl"), "rb") as f:
            feature_indices_list = pickle.load(f) 

for data_files in os.listdir("dataset"):
     if data_files not in features_extract_linkage_all_layers_to_non_duplicate_2:
         continue  

     for prompt_no in range(0, 20):
          if prompt_no not in features_extract_linkage_all_layers_to_non_duplicate_2[data_files]: 
               continue   
          
          user_prompt = benign[data_files][prompt_no]['new'][11:]

          for lyr in range(0, 32):  

             if lyr not in features_extract_linkage_all_layers_to_non_duplicate_2[data_files][prompt_no]:
                  continue
             
             visited = set()
             m = token_ref[data_files][prompt_no]
             visited = feature_indices_list[data_files][prompt_no][lyr][m]

             for feature in visited:
                if feature not in features_extract_linkage_all_layers_to_non_duplicate_2[data_files][prompt_no][lyr]:
                     continue
                c = 1
                f = []
                en_ft = features_extract_linkage_all_layers_to_non_duplicate_2[data_files][prompt_no][lyr][feature]
                for en in range(len(en_ft)):
                    if c <= top_f:
                        s = en_ft[en]
                        f.append(s[2])
                        c += 1
                if f:        
                    print(f"features for layer {lyr} and {feature} are :: ", f)
                    safety_check(f, lyr, feature, user_prompt, "Single-token driven", data_files, prompt_no)        

print("Evaluation 3 = ", evaluations) 

with open("key_details_benign_3.pkl", "wb") as f:
    pickle.dump(key_details, f)