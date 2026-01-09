# type: ignore
import json
import os
import re
import sys
from pathlib import Path

import gradio as gr
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class TrafficVLMDemo:
    def __init__(
        self,
        experiment_name,
        checkpoint_name,
        processed_data_dir="data/processed_5",
        device="cuda",
    ):
        self.experiment_name = experiment_name
        self.processed_data_dir = processed_data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Auto-detect paths
        self.output_dir = Path("outputs") / experiment_name
        self.checkpoint_dir = Path("checkpoints") / experiment_name

        # Add experiment output directory to sys.path FIRST
        sys.path.insert(0, str(self.output_dir))

        # Load configs from outputs directory
        from dataset_config import DatasetConfig  # type: ignore
        from model_config import ModelConfig  # type: ignore

        self.model_config = ModelConfig()
        self.dataset_config = DatasetConfig()

        # Create a fake 'config' module so model imports work
        import types

        config_module = types.ModuleType("config")
        config_module.model_config = types.ModuleType("model_config")
        config_module.model_config.ModelConfig = ModelConfig
        config_module.dataset_config = types.ModuleType("dataset_config")
        config_module.dataset_config.DatasetConfig = DatasetConfig
        sys.modules["config"] = config_module
        sys.modules["config.model_config"] = config_module.model_config
        sys.modules["config.dataset_config"] = config_module.dataset_config

        # Create tokenizer
        self.tokenizer = self._create_tokenizer()

        # Load model
        from model.vlm_model import TrafficVLM

        checkpoint_path = self.checkpoint_dir / checkpoint_name

        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model = TrafficVLM(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Label mapping
        self.id2label = {v: k for k, v in self.dataset_config.label_map.items()}

        print(f"Loaded {checkpoint_name}")
        print(f"Device: {self.device}")
        print(f"Vocab Size: {len(self.tokenizer.vocab)}")
        print(f"Label Map: {self.dataset_config.label_map}")

        # Clean up sys.path
        sys.path.remove(str(self.output_dir))

    def _create_tokenizer(self):
        """Create tokenizer matching the training code"""

        class CustomTokenizer:
            def __init__(self, vocab_path):
                with open(vocab_path, "r") as f:
                    self.vocab = json.load(f)
                self.inverse_vocab = {v: k for k, v in self.vocab.items()}

            def encode(self, text, max_len=None):
                text = text.lower()
                text = re.sub(r"[^\w\s]", "", text)
                words = text.split()
                ids = [self.vocab.get(w, self.vocab.get("[UNK]", 3)) for w in words]
                ids = [self.vocab.get("[SOS]", 1)] + ids + [self.vocab.get("[EOS]", 2)]

                if max_len:
                    if len(ids) > max_len:
                        ids = ids[:max_len]
                        ids[-1] = self.vocab.get("[EOS]", 2)
                    else:
                        ids = ids + [self.vocab.get("[PAD]", 0)] * (max_len - len(ids))
                return ids

        vocab_path = Path(self.processed_data_dir) / "vocab.json"
        return CustomTokenizer(vocab_path)

    def preprocess_image(self, image):
        """Preprocess image matching training"""
        image = image.resize((self.model_config.image_size, self.model_config.image_size))

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.dataset_config.mean, std=self.dataset_config.std
                ),
            ]
        )

        return transform(image).unsqueeze(0)

    def predict(self, image):
        """Run inference with ONLY safety question"""
        if image is None:
            return {"Warning: No Image": 1.0}, "No image provided"

        # FIXED QUESTION - matches training exactly
        question = "Can I move forward?"

        print(f"\n{'=' * 60}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Question: '{question}'")

        # Preprocess
        pixel_values = self.preprocess_image(image).to(self.device)
        input_ids_list = self.tokenizer.encode(
            question, max_len=self.dataset_config.max_seq_len
        )
        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)

        print(f"Tokens: {input_ids_list}")
        print(
            f"Token words: {[self.tokenizer.inverse_vocab.get(t, f'?{t}') for t in input_ids_list]}"
        )

        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values, input_ids)

            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            print(f"Raw logits: {logits[0].cpu().numpy()}")
            probs = F.softmax(logits, dim=-1)[0]
            print(f"Probabilities: {probs.cpu().numpy()}")

        # Map to actual label names from dataset config
        results = {}
        debug_text = "Raw Logits:\n"

        for class_id in range(self.model_config.num_classes):
            label_name = self.id2label.get(class_id, f"class_{class_id}")

            # Format label for display
            if label_name == "yes" or label_name == "safe":
                display_name = "SAFE"
            elif label_name == "no":
                display_name = "NO (generic)"
            elif label_name == "stop_red_light":
                display_name = "STOP: Red Light"
            elif label_name == "stop_pedestrian":
                display_name = "STOP: Pedestrian"
            elif label_name == "stop_vehicle":
                display_name = "STOP: Vehicle"
            elif label_name == "stop_obstacle":
                display_name = "STOP: Obstacle"
            else:
                display_name = label_name.upper()

            logit_val = logits[0][class_id].item()
            prob_val = probs[class_id].item()

            results[display_name] = float(prob_val)
            debug_text += f"{display_name:20s}: logit={logit_val:7.2f}, prob={prob_val * 100:6.2f}%\n"

        print(debug_text)
        print(f"{'=' * 60}\n")

        return results, debug_text


def get_available_experiments():
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return []

    experiments = []
    for exp_dir in outputs_dir.iterdir():
        if exp_dir.is_dir() and (exp_dir / "dataset_config.py").exists():
            experiments.append(exp_dir.name)

    return sorted(experiments)


def get_available_checkpoints(experiment_name):
    checkpoint_dir = Path("checkpoints") / experiment_name

    if not checkpoint_dir.exists():
        return []

    checkpoints = []

    if (checkpoint_dir / "best_model.pt").exists():
        checkpoints.append("best_model.pt")

    epoch_ckpts = sorted(
        checkpoint_dir.glob("checkpoint_epoch_*.pt"),
        key=lambda x: int(x.stem.split("_")[-1]),
    )
    for ckpt in epoch_ckpts:
        checkpoints.append(ckpt.name)

    return checkpoints


def get_available_processed_dirs():
    data_dir = Path("data")

    if not data_dir.exists():
        return []

    processed_dirs = []
    for item in data_dir.iterdir():
        if (
            item.is_dir()
            and item.name.startswith("processed")
            and (item / "vocab.json").exists()
        ):
            processed_dirs.append(str(item))

    return sorted(processed_dirs)


def create_demo():
    available_experiments = get_available_experiments()
    available_processed_dirs = get_available_processed_dirs()

    if not available_experiments:
        raise ValueError("No experiments found")

    if not available_processed_dirs:
        raise ValueError("No processed data directories found")

    initial_experiment = available_experiments[0]
    initial_processed_dir = (
        available_processed_dirs[0] if available_processed_dirs else "data/processed_5"
    )
    initial_checkpoints = get_available_checkpoints(initial_experiment)

    if not initial_checkpoints:
        raise ValueError(f"No checkpoints found for {initial_experiment}")

    demo_instances = {}
    instance_key = f"{initial_experiment}_{initial_processed_dir}"
    demo_instances[instance_key] = TrafficVLMDemo(
        initial_experiment, initial_checkpoints[0], initial_processed_dir
    )

    def update_experiment(experiment_name):
        checkpoints = get_available_checkpoints(experiment_name)
        if not checkpoints:
            return gr.update(choices=[], value=None), "Error: No checkpoints found"

        return gr.update(
            choices=checkpoints, value=checkpoints[0]
        ), f"Loaded {checkpoints[0]}"

    def update_model(experiment_name, checkpoint_name, processed_dir):
        instance_key = f"{experiment_name}_{processed_dir}"
        demo_instances[instance_key] = TrafficVLMDemo(
            experiment_name, checkpoint_name, processed_dir
        )
        return f"Loaded {checkpoint_name} with {processed_dir}"

    def predict_wrapper(image, experiment, checkpoint, processed_dir):
        instance_key = f"{experiment}_{processed_dir}"
        if instance_key not in demo_instances:
            demo_instances[instance_key] = TrafficVLMDemo(
                experiment, checkpoint, processed_dir
            )
        return demo_instances[instance_key].predict(image)

    with gr.Blocks(title="Traffic VLM - Safety Demo") as interface:
        gr.Markdown("""
        # Traffic VLM - Safety Assessment
        
        **Question (Fixed)**: "Can I move forward?"
        
        Upload a traffic image to get safety prediction.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                experiment_selector = gr.Dropdown(
                    choices=available_experiments,
                    value=initial_experiment,
                    label="Experiment",
                    interactive=True,
                )

                checkpoint_selector = gr.Dropdown(
                    choices=initial_checkpoints,
                    value=initial_checkpoints[0],
                    label="Checkpoint",
                    interactive=True,
                )

                processed_dir_selector = gr.Dropdown(
                    choices=available_processed_dirs,
                    value=initial_processed_dir,
                    label="Processed Data Dir",
                    interactive=True,
                )

                model_status = gr.Textbox(
                    value=f"Loaded {initial_checkpoints[0]}",
                    label="Status",
                    interactive=False,
                )

                image_input = gr.Image(type="pil", label="Upload Traffic Image")

                submit_btn = gr.Button("🚦 Assess Safety", variant="primary", size="lg")

            with gr.Column(scale=1):
                output = gr.Label(num_top_classes=6, label="Safety Prediction")

                debug_output = gr.Textbox(
                    label="Debug Info (Logits & Probabilities)",
                    lines=12,
                    interactive=False,
                )

                gr.Markdown("""
                ### Expected Output Classes:
                - **SAFE** → Clear to proceed
                - **NO (generic)** → Stop (unspecified)
                - **STOP: Red Light** → Traffic light is red
                - **STOP: Pedestrian** → Pedestrian detected
                - **STOP: Vehicle** → Vehicle blocking
                - **STOP: Obstacle** → Other obstacle
                
                ---
                
                **⚠️ Training Issue Diagnosis:**
                - If logits for classes 2-5 are < -7, model collapsed during training
                - All "STOP" classes showing 0% means model never learned specific stop reasons
                - Try different checkpoints (early epochs may work better)
                - Check training data class balance
                """)

        experiment_selector.change(
            fn=update_experiment,
            inputs=[experiment_selector],
            outputs=[checkpoint_selector, model_status],
        )

        checkpoint_selector.change(
            fn=update_model,
            inputs=[experiment_selector, checkpoint_selector, processed_dir_selector],
            outputs=[model_status],
        )

        processed_dir_selector.change(
            fn=update_model,
            inputs=[experiment_selector, checkpoint_selector, processed_dir_selector],
            outputs=[model_status],
        )

        submit_btn.click(
            fn=predict_wrapper,
            inputs=[
                image_input,
                experiment_selector,
                checkpoint_selector,
                processed_dir_selector,
            ],
            outputs=[output, debug_output],
        )

    return interface


if __name__ == "__main__":
    print("=" * 60)
    print("TRAFFIC VLM - SAFETY ASSESSMENT DEMO")
    print("=" * 60)

    if not Path("outputs").exists():
        print("Error: outputs directory not found!")
        exit(1)

    experiments = get_available_experiments()
    if not experiments:
        print("Error: No experiments found!")
        exit(1)

    processed_dirs = get_available_processed_dirs()
    print(f"\nProcessed dirs: {len(processed_dirs)}")
    print(f"Experiments: {len(experiments)}")

    for exp in experiments[:3]:  # Show first 3
        checkpoints = get_available_checkpoints(exp)
        print(f"\n  {exp}: {len(checkpoints)} checkpoints")

    print(f"\n{'=' * 60}\n")

    demo = create_demo()
    demo.launch(share=True, server_port=7860, server_name="0.0.0.0")
