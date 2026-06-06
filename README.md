# Mechanistically_Steering

This repository contains the codes required as part of the Student Research Workshop.

## Order of Running

1. Modify config (`baselines/model_config.py`).
2. Run attack (`baselines/run_attack.py`).

   Before running attack, set up `sae_robustness` and patch its loader:
   1. Clone the repo:
      ```bash
      git clone https://github.com/AI4LIFE-GROUP/sae_robustness
      ```
   2. Open `sae_robustness/src/loader.py` and make these edits:
      - Replace line 31 and line 32 with:
        ```python
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", cache_dir=CACHE_DIR).to(DEVICE)
        ```
      - Remove line 33.
      - Change line 36 to:
        ```python
        			  filename="layer_16/width_16k/average_l0_75/params.npz",
        ```
      - Make sure the `sae_robustness` directory is in the root directory.

3. Run inference (`baselines/run_inference.py`).
4. Fix empty responses (`baselines/fix_empty_responses.py`).
5. Extract results (`baselines/extract_results.py`).
6. Run judge (`baselines/run_judge.py`).
