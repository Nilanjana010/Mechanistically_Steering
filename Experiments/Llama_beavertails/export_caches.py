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
    n_devices=2,
)

model.embed.to("cuda:0")

for layer in range(32):
    if layer < 16:
        model.blocks[layer].to("cuda:0")
    else:
        model.blocks[layer].to("cuda:1")

model.ln_final.to("cuda:1")
model.unembed.to("cuda:1")

# CHECK WHERE THE MODEL WAS LOADED
print("\n===== MODEL DEVICE PLACEMENT =====")

print("Embedding:", model.embed.W_E.device)

for i in range(32):
    print(
        f"Block {i}:",
        {str(p.device) for p in model.blocks[i].parameters()}
    )

print("Unembed:", model.unembed.W_U.device)

folder = "dataset"
all_data = None

folder_pos = "exported_caches"
os.makedirs(folder_pos, exist_ok=True)

folder_pos_20  = "exported_caches_20"
os.makedirs(folder_pos_20, exist_ok=True)

sae_list = []

for lyr_num in range(32):

    idx_sae_1 = f"l{lyr_num}r_8x"
    print("idx_sae_1 is == ", idx_sae_1)

    # Find where this model block actually lives
    layer_device = str(
        next(model.blocks[lyr_num].parameters()).device
    )

    print(
        f"Layer {lyr_num} is on {layer_device}; "
        f"loading SAE on {layer_device}"
    )

    sae, cfg_dict, sparsity = (
        SAE.from_pretrained_with_cfg_and_sparsity(
            release="llama_scope_lxr_8x",
            sae_id=idx_sae_1,
            device=layer_device,
        )
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

            embed_device = model.embed.W_E.device

            tokens = model.to_tokens(all_data[i]['prompt']).to(embed_device)

            print("Embedding device =", embed_device)
            print("Tokens device    =", tokens.device)

            _, cache = model.run_with_cache_with_saes(tokens, saes=sae_list)
            
            print("Cache keys:")
            for key in cache.keys():
                print(key)

            for layer in range(0, 32):
                tensor_name = f"cache_tensor_{filename}_prompt_{i}_layer_{layer}.pt"
                print(f"tensor_name = {tensor_name}")
                tensor_path = os.path.join(os.path.join(folder_pos, cache_fol), tensor_name)
                lyr_hook = "blocks." + str(layer) + ".hook_resid_post.hook_sae_acts_post" # eg., blocks.0.attn.hook_z.hook_sae_acts_post
                print("layer hook == ", lyr_hook)
                torch.save(cache[lyr_hook].detach().cpu(), tensor_path)

            tensor_20_name = f"cache_tensor_20_{filename}_prompt_{i}.pt"
            print(f"tensor_20_name = {tensor_20_name}")
            tensor_20_path = os.path.join(folder_pos_20, tensor_20_name)
            torch.save(cache["blocks.20.hook_resid_post.hook_sae_input"].detach().cpu(), tensor_20_path) 