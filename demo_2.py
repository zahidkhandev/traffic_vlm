# type: ignore
import importlib
import json
import os
import re
import sys
import traceback
import types
from pathlib import Path

import gradio as gr
import matplotlib.cm as cm
import numpy as np
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

        self.output_dir = Path("outputs") / experiment_name
        self.checkpoint_dir = Path("checkpoints") / experiment_name

        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint_dir}"
            )

        sys.path.insert(0, str(self.output_dir))

        try:
            if "dataset_config" in sys.modules:
                del sys.modules["dataset_config"]
            if "model_config" in sys.modules:
                del sys.modules["model_config"]

            from dataset_config import DatasetConfig
            from model_config import ModelConfig

            self.model_config = ModelConfig()
            self.dataset_config = DatasetConfig()

            config_module = types.ModuleType("config")
            config_module.model_config = types.ModuleType("model_config")
            config_module.model_config.ModelConfig = ModelConfig
            config_module.dataset_config = types.ModuleType("dataset_config")
            config_module.dataset_config.DatasetConfig = DatasetConfig

            sys.modules["config"] = config_module
            sys.modules["config.model_config"] = config_module.model_config
            sys.modules["config.dataset_config"] = config_module.dataset_config

            if "model.vlm_model" in sys.modules:
                importlib.reload(sys.modules["model.vlm_model"])

            from model.vlm_model import TrafficVLM

        finally:
            if str(self.output_dir) in sys.path:
                sys.path.remove(str(self.output_dir))

        self.tokenizer = self._create_tokenizer()

        checkpoint_path = self.checkpoint_dir / checkpoint_name
        print(f"Loading model from {checkpoint_path}...")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = TrafficVLM(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.id2label = {v: k for k, v in self.dataset_config.label_map.items()}
        self.captured_attn = None

        print(f"Loaded {checkpoint_name}")
        print(f"Device: {self.device}")
        print(f"Vocab Size: {len(self.tokenizer.vocab)}")

    def _create_tokenizer(self):
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
        if not vocab_path.exists():
            print(f"Warning: Vocab not found at {vocab_path}")

        return CustomTokenizer(vocab_path)

    def preprocess_image(self, image):
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

    def generate_attention_overlay(self, image, attn_weights):
        """
        Robust attention overlay generator.
        Handles:
        1. Built-in method output: (Batch, Seq, H, W)
        2. Hook output: (Batch, Heads, Target, Source) or (Target, Source)
        3. CLS token stripping
        """
        if attn_weights is None:
            print("Visualizer: No attention weights captured.")
            return image

        try:
            if isinstance(attn_weights, torch.Tensor) and attn_weights.dim() == 4:
                attn_map = attn_weights[0, -1, :, :].detach().cpu().numpy()
                print("Visualizer: Using Direct Map (Batch, Seq, H, W)")

            else:
                if isinstance(attn_weights, tuple):
                    attn_weights = attn_weights[1]

                if attn_weights.dim() == 4 and attn_weights.shape[0] == 1:
                    attn_weights = attn_weights[0]  # (Heads, Target, Source)

                if attn_weights.dim() == 3:
                    attn_weights = attn_weights.mean(dim=0)

                if attn_weights.dim() == 2:
                    attn_weights = attn_weights[-1]

                num_patches = attn_weights.shape[-1]

                grid_size = int(np.sqrt(num_patches))
                if grid_size * grid_size != num_patches:
                    grid_size_minus_1 = int(np.sqrt(num_patches - 1))
                    if grid_size_minus_1 * grid_size_minus_1 == num_patches - 1:
                        print("Visualizer: Detected CLS token. Dropping first token.")
                        attn_weights = attn_weights[1:]
                        grid_size = grid_size_minus_1
                    else:
                        print(
                            f"Visualizer Error: Cannot reshape attention map of size {num_patches}"
                        )
                        return image

                attn_map = (
                    attn_weights.reshape(grid_size, grid_size).detach().cpu().numpy()
                )
                print(
                    f"Visualizer: Reconstructed Map from Hook ({grid_size}x{grid_size})"
                )

            attn_map = (attn_map - attn_map.min()) / (
                attn_map.max() - attn_map.min() + 1e-8
            )

            heatmap = Image.fromarray(np.uint8(255 * attn_map))
            heatmap = heatmap.resize(image.size, resample=Image.BILINEAR)

            heatmap_np = np.array(heatmap)

            colormap = cm.get_cmap("jet")
            colored_heatmap = colormap(heatmap_np / 255.0)
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            colored_heatmap_img = Image.fromarray(colored_heatmap)

            return Image.blend(image, colored_heatmap_img, alpha=0.5)

        except Exception as e:
            print(f"Visualizer Exception: {e}")
            traceback.print_exc()
            return image

    def predict(self, image):
        if image is None:
            return {"Warning: No Image": 1.0}, "No image provided", None

        question = "Can I move forward?"

        pixel_values = self.preprocess_image(image).to(self.device)
        input_ids_list = self.tokenizer.encode(
            question, max_len=self.dataset_config.max_seq_len
        )
        input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)

        self.captured_attn = None
        hook_handles = []

        use_method = hasattr(self.model, "get_cross_attention_map")

        if not use_method:
            print("[System] 'get_cross_attention_map' not found. Using Hooks.")

            def hook_fn(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    self.captured_attn = output[1]
                elif isinstance(output, torch.Tensor) and output.dim() == 4:
                    self.captured_attn = output

            for name, module in self.model.named_modules():
                if "cross_attn" in name or (
                    "multihead_attn" in name and "decoder" in name
                ):
                    hook_handles.append(module.register_forward_hook(hook_fn))

        with torch.no_grad():
            if use_method:
                outputs = self.model(pixel_values, input_ids)
                self.captured_attn = self.model.get_cross_attention_map(
                    pixel_values, input_ids
                )
            else:
                outputs = self.model(pixel_values, input_ids)

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = F.softmax(logits, dim=-1)[0]

        for h in hook_handles:
            h.remove()

        results = {}
        debug_text = f"Experiment: {self.experiment_name}\n"
        debug_text += f"Question: '{question}'\nTokens: {input_ids_list}\n"
        debug_text += f"Viz Method: {'Built-in' if use_method else 'Hook'}\n\nLogits:\n"

        for class_id in range(self.model_config.num_classes):
            label_name = self.id2label.get(class_id, f"class_{class_id}")

            if label_name in ["yes", "safe"]:
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

            prob_val = probs[class_id].item()
            results[display_name] = float(prob_val)
            debug_text += f"{display_name:20s}: {prob_val * 100:6.2f}%\n"

        attn_overlay = self.generate_attention_overlay(image, self.captured_attn)

        print(debug_text)
        print("-" * 40)
        return results, debug_text, attn_overlay


def get_available_experiments():
    checkpoints_root = Path("checkpoints")
    outputs_root = Path("outputs")

    if not checkpoints_root.exists() or not outputs_root.exists():
        return []

    experiments = set()
    for item in checkpoints_root.iterdir():
        if item.is_dir():
            config_path = outputs_root / item.name / "dataset_config.py"
            if config_path.exists():
                experiments.add(item.name)

    def sort_key(name):
        v_match = re.search(r"v(\d+)", name)
        v_num = int(v_match.group(1)) if v_match else 0
        run_match = re.search(r"run_(\d+)", name)
        run_num = int(run_match.group(1)) if run_match else 0
        is_polish = 1 if "polish" in name else 0
        is_finetune = 1 if "finetune" in name else 0
        return (v_num, run_num, is_polish, is_finetune, name)

    return sorted(list(experiments), key=sort_key, reverse=True)


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
        reverse=True,
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
    return sorted(processed_dirs, reverse=True)


def create_demo():
    experiments = get_available_experiments()
    processed_dirs = get_available_processed_dirs()

    if not experiments:
        raise ValueError("No valid experiments found in checkpoints/ & outputs/")

    default_exp = experiments[0]
    default_processed = processed_dirs[0] if processed_dirs else "data/processed_5"
    default_ckpts = get_available_checkpoints(default_exp)
    default_ckpt = default_ckpts[0] if default_ckpts else None

    demo_cache = {}

    def get_predictor(experiment, checkpoint, processed_dir):
        key = f"{experiment}|{checkpoint}|{processed_dir}"
        if key not in demo_cache:
            demo_cache.clear()
            print(f"[System] Loading new model: {key}")
            demo_cache[key] = TrafficVLMDemo(experiment, checkpoint, processed_dir)
        return demo_cache[key]

    def update_ckpt_list(exp_name):
        ckpts = get_available_checkpoints(exp_name)
        new_val = ckpts[0] if ckpts else None
        return gr.update(choices=ckpts, value=new_val), f"Selected {exp_name}"

    def run_prediction(image, exp, ckpt, p_dir):
        if not image or not exp or not ckpt:
            return None, "Missing inputs", None
        try:
            predictor = get_predictor(exp, ckpt, p_dir)
            return predictor.predict(image)
        except Exception as e:
            return None, f"Error: {str(e)}", None

    with gr.Blocks(title="Traffic VLM Safety Inspector") as app:
        gr.Markdown("#Traffic VLM Safety Inspector")
        gr.Markdown("Upload an image to assess if it is safe to move forward.")

        with gr.Row():
            with gr.Column(scale=1):
                dd_exp = gr.Dropdown(
                    label="Experiment", choices=experiments, value=default_exp
                )
                dd_ckpt = gr.Dropdown(
                    label="Checkpoint", choices=default_ckpts, value=default_ckpt
                )
                dd_data = gr.Dropdown(
                    label="Data Dir", choices=processed_dirs, value=default_processed
                )

                status_txt = gr.Textbox(
                    label="System Status", value="Ready", interactive=False
                )

                img_input = gr.Image(label="Dashboard Camera View", type="pil")
                btn_predict = gr.Button("Analyze Safety", variant="primary")

            with gr.Column(scale=1):
                lbl_result = gr.Label(label="Safety Decision", num_top_classes=5)
                txt_debug = gr.Textbox(label="Internal Confidence (Logits)", lines=10)
                img_attn = gr.Image(label="Cross-Attention Visualization", type="pil")

        dd_exp.change(fn=update_ckpt_list, inputs=[dd_exp], outputs=[dd_ckpt, status_txt])

        btn_predict.click(
            fn=run_prediction,
            inputs=[img_input, dd_exp, dd_ckpt, dd_data],
            outputs=[lbl_result, txt_debug, img_attn],
        )

    return app


if __name__ == "__main__":
    print("Starting Traffic VLM Demo...")
    try:
        demo = create_demo()
        demo.launch(share=True, server_port=7860, server_name="0.0.0.0")
    except Exception as e:
        print(f"Failed to launch demo: {e}")
