"""
V15 Student Model — MLflow PythonModel wrapper.

Wraps the Qwen3-4B bf16 + LoRA adapter for inference via MLflow.
Loads the model once, serves predictions via mlflow.pyfunc.

Usage:
    # Register:
    python3 finetune/mlflow_model_v15.py

    # Then load and predict:
    model = mlflow.pyfunc.load_model("models:/v15-student/production")
    result = model.predict({"title": "...", "location": "...", "jd_text": "..."})
"""

import json
import os
import sys

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types import ColSpec, DataType, ParamSchema, ParamSpec, Schema


class V15StudentModel(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for V15 Qwen3-4B LoRA student model."""

    def load_context(self, context):
        """Load model + adapter + prompt on first call."""
        from mlx_lm import load, generate

        self._generate = generate

        # Read config
        config_path = context.artifacts["model_config"]
        with open(config_path) as f:
            config = json.load(f)

        model_name = config["model_name"]
        adapter_dir = config["adapter_dir"]

        print(f"Loading model: {model_name}")
        print(f"Loading adapter: {adapter_dir}")

        self.model, self.tokenizer = load(model_name, adapter_path=adapter_dir)

        # Load prompt template
        with open(context.artifacts["prompt_path"]) as f:
            self.prompt_template = f.read()

        print(f"Prompt loaded: {len(self.prompt_template)} chars")
        print("V15 Student Model ready.")

    def _build_prompt(self, title, location, jd_text):
        """Fill in the prompt template with job data."""
        prompt = self.prompt_template
        prompt = prompt.replace("{{job_title}}", title or "")
        prompt = prompt.replace("{{job_location}}", location or "")
        prompt = prompt.replace("{{jd_text}}", jd_text or "")
        return prompt

    def predict(self, context, model_input, params=None):
        """Run inference on one or more jobs.

        model_input: DataFrame with columns [title, location, jd_text]
        Returns: DataFrame with columns [response, tokens]
        """
        max_tokens = (params or {}).get("max_tokens", 500)

        results = []
        for _, row in model_input.iterrows():
            title = str(row.get("title", ""))
            location = str(row.get("location", ""))
            jd_text = str(row.get("jd_text", ""))

            user_msg = self._build_prompt(title, location, jd_text)

            messages = [
                {"role": "system", "content": "Respond with JSON only."},
                {"role": "user", "content": user_msg},
            ]

            prompt_tokens = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            # Pre-fill with { for JSON
            prompt_tokens += "{"

            response = self._generate(
                self.model, self.tokenizer,
                prompt=prompt_tokens,
                max_tokens=max_tokens,
                verbose=False,
            )

            # Parse response
            raw = "{" + response
            try:
                parsed = json.loads(raw)
                tokens = {
                    "loc": parsed.get("loc"),
                    "arr": parsed.get("arr"),
                    "sen": parsed.get("sen"),
                    "tech": parsed.get("tech"),
                    "comp": parsed.get("comp"),
                }
            except json.JSONDecodeError:
                tokens = {"error": "parse_fail", "raw": raw[:200]}

            results.append({
                "response": raw,
                "tokens": json.dumps(tokens),
            })

        return pd.DataFrame(results)


def register_model():
    """Register the V15 model with MLflow."""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("V15-Student-Model")

    input_schema = Schema([
        ColSpec(DataType.string, "title"),
        ColSpec(DataType.string, "location"),
        ColSpec(DataType.string, "jd_text"),
    ])
    output_schema = Schema([
        ColSpec(DataType.string, "response"),
        ColSpec(DataType.string, "tokens"),
    ])
    parameters = ParamSchema([
        ParamSpec("max_tokens", DataType.integer, np.int32(500), None),
    ])
    signature = ModelSignature(
        inputs=input_schema, outputs=output_schema, params=parameters,
    )

    input_example = pd.DataFrame([{
        "title": "Senior Software Engineer",
        "location": "London, UK",
        "jd_text": "We are looking for a Senior Software Engineer with Node.js and React experience.",
    }])

    # Write model config to a temp file (MLflow artifacts must be local files)
    import tempfile, os
    config_dir = tempfile.mkdtemp(prefix="mlflow_v15_")
    config_path = os.path.join(config_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": "mlx-community/Qwen3-4B-bf16",
            "adapter_dir": os.path.abspath("finetune/adapters_v15_4B"),
            "prompt_path": os.path.abspath("prompts/student_v15_fix4.txt"),
        }, f)

    with mlflow.start_run(run_name="v15-model-registration"):
        model_info = mlflow.pyfunc.log_model(
            name="v15-student",
            python_model=V15StudentModel(),
            artifacts={
                "model_config": config_path,
                "prompt_path": os.path.abspath("prompts/student_v15_fix4.txt"),
            },
            pip_requirements=[
                "mlx>=0.31",
                "mlx-lm>=0.31",
                "transformers>=5.0",
            ],
            input_example=input_example,
            signature=signature,
        )

        mlflow.log_params({
            "model": "mlx-community/Qwen3-4B-bf16",
            "adapter": "finetune/adapters_v15_4B/0000700_adapters.safetensors",
            "prompt": "student_v15_fix4.txt",
            "model_only_accuracy": 91.4,
            "hybrid_accuracy": 99.6,
        })

        print(f"\nModel registered: {model_info.model_uri}")
        print(f"Load with: mlflow.pyfunc.load_model('{model_info.model_uri}')")

    # Set production alias
    from mlflow.tracking import MlflowClient
    client = MlflowClient("http://localhost:5000")

    # Get the latest model version
    versions = client.search_model_versions(f"name='v15-student'")
    if versions:
        latest = max(versions, key=lambda v: int(v.version))
        client.set_registered_model_alias("v15-student", "production", latest.version)
        print(f"Alias 'production' → v{latest.version}")

    print("\nDone! Load model with:")
    print('  model = mlflow.pyfunc.load_model("models:/v15-student@production")')
    print('  result = model.predict(pd.DataFrame([{"title": "...", "location": "...", "jd_text": "..."}]))')


if __name__ == "__main__":
    register_model()
