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

model = HookedSAETransformer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device="cuda",
)

sae_device = str(next(model.blocks[20].parameters()).device)
print("sae_device = ", sae_device)

sae_20, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
    release="llama_scope_lxr_8x",
    sae_id="l20r_8x",
    device=sae_device,
)

print("Layer 20 device:", sae_device)
print("SAE device:", next(sae_20.parameters()).device)

for i in [0, 10, 20, 31]:
    print(i, next(model.blocks[i].parameters()).device)

# Print configuration
print("sae_20.cfg.__dict__ =", sae_20.cfg.__dict__)

# Preserve the original reconstruction error when the SAE
# is inserted into the model's forward pass
sae_20.use_error_term = True

print("use_error_term =", sae_20.use_error_term)

pkl_dir = "concept_dictionaries"

concept_vector_Folder = "concept_vector_dictionaries"
os.makedirs(concept_vector_Folder, exist_ok=True)

all_pkls = {}

for fname in os.listdir(pkl_dir):
        file_path = os.path.join(pkl_dir, fname)
        print(f"Loading {fname} ...")
        with open(file_path, "rb") as f:
            all_pkls[fname] = pickle.load(f)

        concept_vector_fname_Folder = f"concept_vector_{fname}_dictionaries"
        os.makedirs(os.path.join(concept_vector_Folder, concept_vector_fname_Folder), exist_ok=True)

        for l in range(len(all_pkls[fname])):  
            print(f"all_pkls[{fname}][{l}][0] = ", all_pkls[fname][l][0])  
            concepts = all_pkls[fname][l][0][11:]
            print("concepts here = ", concepts)
            if concepts:
                items = [x.strip() for x in concepts.split(",")]
                print("items here are = ", items)
                for item in items:
                        your_new_concept = f"{item}"
                        print(f"your_new_concept = {your_new_concept}")

                        _, cache = model.run_with_cache_with_saes(
                            your_new_concept,
                            saes=sae_20
                        )

                        tokens = model.to_str_tokens(your_new_concept)

                        print("Tokens:")
                        for i, token in enumerate(tokens):
                            print(i, repr(token))
                            
                        for key in cache.keys():
                            print("Keys are = ", key)

                        acts = cache['blocks.20.hook_resid_post.hook_sae_input'] 
                        print("Dimensions of acts = ", acts, "shape = ", acts.shape)

                        sp = acts[0, 1:, :].mean(dim=0)
                        print("concept vector:", sp.shape, "acts[0, 1:, :].shape", acts[0, 1:, :].shape)
                    
                        print(f"subspace for {item} = ", sp)

                        tensor_name = f"tensor_{item}_prompt_{l}.pt"
                        print(f"tensor_name = {tensor_name}")
                        tensor_path = os.path.join(os.path.join(concept_vector_Folder, concept_vector_fname_Folder), tensor_name)
                        torch.save(sp.detach().cpu(), tensor_path)