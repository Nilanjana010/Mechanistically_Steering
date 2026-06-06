# Step 1: Choose target model size.
TARGET_MODEL_SIZE = "9b"

# Step 2: Define config values per model size.
MODEL_CONFIG = {
    "2b": {
        "model_id": "google/gemma-2-2b",
        "model_type": "gemma2-2b-16k",
        "results_prefix": "gemma2-2b-16k-",
    },
    "9b": {
        "model_id": "google/gemma-2-9b-it",
        "model_type": "gemma2-9b-16k",
        "results_prefix": "gemma2-9b-16k-",
    },
}


def get_model_config(target_model_size: str = TARGET_MODEL_SIZE) -> dict:
    # Step 3: Validate and return model config.
    if target_model_size not in MODEL_CONFIG:
        raise ValueError(
            f"Unsupported model size '{target_model_size}'. "
            f"Choose from: {list(MODEL_CONFIG.keys())}"
        )
    return MODEL_CONFIG[target_model_size]


def get_model_id(target_model_size: str = TARGET_MODEL_SIZE) -> str:
    # Step 4: Return Hugging Face model id.
    return get_model_config(target_model_size)["model_id"]


def get_model_type(target_model_size: str = TARGET_MODEL_SIZE) -> str:
    # Step 5: Return attack model type string.
    return get_model_config(target_model_size)["model_type"]


def get_results_prefix(target_model_size: str = TARGET_MODEL_SIZE) -> str:
    # Step 6: Return result folder prefix.
    return get_model_config(target_model_size)["results_prefix"]