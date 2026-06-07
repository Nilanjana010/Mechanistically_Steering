# Mechanistically_Steering
## Gemma-2-2B/Gemma-2-9B:

1. To create the dataset from BeaverTails run `create_dataset.py`.
2. To extract negative sentiments run `final_entity_grok.py`.
3. To generate concept subspaces run `concept_generator.py`.
4. To extract the activation caches run `export_caches.py`.
5. To compute cosine similarities run `gemma.py`.
6. To execute the three feature-grouping strategies run `agglo_updated_linkage_2.py` and `agglo_updated_linkage_3.py`.
7. To create benign looking versions of the original prompts run `grok_benign.py`.
8. To execute the steering experiments run `feature_steer_3.py` and `feature_steer_benign*.py`.
9. To evaluate responses using Grok run `grok_jd.py` and `grok_jd_benign*.py`.
10. For plots refer to `results*.ipynb`.

## Baseline:

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
