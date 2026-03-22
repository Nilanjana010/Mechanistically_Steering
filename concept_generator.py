import torch
import pickle
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class RegressionWrapper(torch.nn.Module):
    def __init__(self, base_model, hidden_size, output_dim):
        super().__init__()
        self.base_model = base_model
        self.regression_head = torch.nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hiddens = outputs.hidden_states[-1]
        last_token_representations = last_hiddens[:, -1]
        preds = self.regression_head(last_token_representations)
        preds = F.normalize(preds, p=2, dim=-1)
        return preds

base_model = AutoModelForCausalLM.from_pretrained(
    f"google/gemma-2-2b", torch_dtype=torch.bfloat16)
base_tokenizer = AutoTokenizer.from_pretrained(
    f"google/gemma-2-2b", model_max_length=512)

hidden_size = base_model.config.hidden_size

# Load the saved state dict
state_dict = torch.load('weight-gemma-reft-2b-it-res-generator.pt', map_location="cpu")

output_dim = state_dict['regression_head.weight'].shape[0]
print("hidden_size = ", hidden_size)
print("hidden = ", state_dict['regression_head.weight'].shape[1])
print("output_dim   = ", output_dim)

subspace_gen = RegressionWrapper(
    base_model, hidden_size, output_dim).bfloat16().to("cuda")
subspace_gen.load_state_dict(torch.load('weight-gemma-reft-2b-it-res-generator.pt'))

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
                        your_new_concept = f"terms related to {item}"
                        print(f"your_new_concept = {your_new_concept}")
                        inputs = base_tokenizer(your_new_concept, return_tensors="pt").to("cuda")
                        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
                        sp = subspace_gen(input_ids, attention_mask)[0]

                        print(f"subspace for {item} = ", sp)
                        print(f"shape of subspace for {item} = ", sp.shape)

                        tensor_name = f"tensor_{item}_prompt_{l}.pt"
                        print(f"tensor_name = {tensor_name}")
                        tensor_path = os.path.join(os.path.join(concept_vector_Folder, concept_vector_fname_Folder), tensor_name)
                        torch.save(sp, tensor_path)