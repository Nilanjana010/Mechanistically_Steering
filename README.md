## Gemma-2-2B/Gemma-2-9B-IT:

1. To create the dataset from BeaverTails run `create_dataset.py`.
2. To extract negative sentiments run `final_entity_grok.py`.
3. To generate concept subspaces run `concept_generator.py`. Download the subspace generators used in this research, introduced by Wu et al. (2025), from Hugging Face:  
[Gemma-ReFT-2B-IT](https://huggingface.co/pyvene/gemma-reft-2b-it-res-generator) / [Gemma-ReFT-9B-IT](https://huggingface.co/pyvene/gemma-reft-9b-it-res-generator).
4. To extract the activation caches run `export_caches.py`.
5. To compute cosine similarities run `gemma.py`.
6. To execute the three feature-grouping strategies run `agglo_updated_linkage_2.py` and `agglo_updated_linkage_3.py`.
7. To create benign looking versions of the original prompts run `grok_benign.py`.
8. To execute the steering experiments run `feature_steer_3.py` and `feature_steer_benign*.py`.
9. To evaluate responses using Grok run `grok_jd.py` and `grok_jd_benign*.py`.
10. For plots refer to `results*.ipynb`.

## Baseline:

1. Modify config (`baselines/model_config.py`) You can change the TARGET_MODEL_SIZE line to either 2b or 9b to run the respective models. Keep the same throughout entire run of the baseline files.
2. Run attack (`baselines/run_attack.py`).

   Before running, set up `sae_robustness` and patch its loader for Gemma 2-9B-it:
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
      - Remove line 33:
        ```python
        assert layer_num == 30
        ```
      - Change line 36 to:
        ```python
        filename="layer_16/width_16k/average_l0_75/params.npz",
        ```
      - This is the final step for Gemma 2-9B-it.
      - Nothing is required for Gemma 2-2B.
      - Make sure the `sae_robustness` directory is in the root directory.
      - This is required to properly run the run_attack python file.

3. Run inference (`baselines/run_inference.py`).
4. Fix empty responses (`baselines/fix_empty_responses.py`).
5. Extract results (`baselines/extract_results.py`).
6. Run judge (`baselines/run_judge.py`).
