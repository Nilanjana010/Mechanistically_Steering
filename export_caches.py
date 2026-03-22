import os
import torch
import pickle
from huggingface_hub import login
from sae_lens import SAE, HookedSAETransformer
login(token="****")

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

print("----------------------------------------------Starts here----------------------------------------------")

model = HookedSAETransformer.from_pretrained("gemma-2-2b", device="cuda")

folder = "dataset"
all_data = None

folder_pos = "exported_caches"
os.makedirs(folder_pos, exist_ok=True)

folder_pos_20  = "exported_caches_20"
os.makedirs(folder_pos_20, exist_ok=True)

sae_list = []
for lyr_num in range(0, 26):
                idx_sae_1 = "layer_" + str(lyr_num) + "/width_16k/canonical"   # eg., layer_0/width_16k/canonical
                print("idx_sae_1 is == ", idx_sae_1)
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                        release="gemma-scope-2b-pt-res-canonical",  # <- Release name
                        sae_id=idx_sae_1,  # <- SAE id (not always a hook point!)
                        device=device,
                    )
                print("sae.cfg.__dict__ = ", sae.cfg.__dict__)
                sae.use_error_term  # If use error term is set to false, we will modify the forward pass by using the sae.
                sae_list.append(sae)
print("sae_list is == ", sae_list)

for filename in os.listdir(folder):
    if filename.endswith(".pkl"):
            path = os.path.join(folder, filename)
            with open(path, "rb") as f:
                all_data = pickle.load(f)

    cache_fol = "cache_fol_" + filename
    os.makedirs(os.path.join(folder_pos, cache_fol), exist_ok=True)

    for i in range(len(all_data)):
            print("\n\n\n\nPrompt:", all_data[i]['prompt'], "\nResponse starts here\n", all_data[i]['response'] + "\n over")  

            _, cache = model.run_with_cache_with_saes(all_data[i]['prompt'], saes=sae_list)
            
            print(f"cache for {filename} and indx {i} is == ", cache)

            for layer in range(0, 26):
                tensor_name = f"cache_tensor_{filename}_prompt_{i}_layer_{layer}.pt"
                print(f"tensor_name = {tensor_name}")
                tensor_path = os.path.join(os.path.join(folder_pos, cache_fol), tensor_name)
                lyr_hook = "blocks." + str(layer) + ".hook_resid_post.hook_sae_acts_post"
                print("layer hook == ", lyr_hook)
                torch.save(cache[lyr_hook], tensor_path) 
              
            tensor_20_name = f"cache_tensor_20_{filename}_prompt_{i}.pt"
            print(f"tensor_20_name = {tensor_20_name}")
            tensor_20_path = os.path.join(folder_pos_20, tensor_20_name)
            torch.save(cache['blocks.20.hook_resid_post.hook_sae_input'], tensor_20_path)  
   
print("----------------------------------------------Ends here----------------------------------------------")