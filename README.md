# Traffic Scene Command Understanding VLM

## 📋 Project Overview

**Goal:** Build a Vision-Language Model (VLM) from scratch that understands traffic scenes through natural language commands. The model answers questions like _"Is the left lane clear?"_ or _"Can I safely turn right?"_ based on visual input.

**Architecture:** **PaliGemma-Inspired**

- **Vision Encoder:** SigLip (Signal Lidar Image Pre-training) style encoder.
- **Language Decoder:** Gemma-style decoder with RoPE embeddings.
- **Fusion:** Linear projection + Cross-Attention mechanism.

**Timeline:** 10 Days (Research-Grade Implementation)

---

## 🛠️ Setup & Installation

### 1. Prerequisites

- **OS:** Windows / Linux / MacOS
- **Python:** 3.8+
- **Hardware:** GPU recommended (NVIDIA RTX 3000 series or better)

### 2. Installation

1. **Clone/Create Project:**
   (If you haven't run the setup script yet, ensure the directory structure matches the blueprint).
2. **Install Dependencies:**

```bash
pip install torch torchvision numpy pandas matplotlib tqdm h5py scikit-learn

```

3. **Data Setup:**

- Download the [BDD100K Dataset](https://www.google.com/search?q=https://bdd-data.berkeley.edu/).
- Place the images in a raw data folder (configure path in `config/dataset_config.py`).

---

## 🗂️ Project Structure

```text
traffic_vlm/
├── config/             # Hyperparameters & Paths
├── data/               # Dataset processing, tokenizers, loaders
├── model/              # Core VLM Architecture
│   ├── vision/         # SigLip Encoder components
│   ├── language/       # Gemma Decoder components
│   └── fusion/         # Cross-modal projection & attention
├── training/           # Training loops, loss, optimizers
├── evaluation/         # Metrics & Inference logic
├── visualization/      # Attention heatmaps & failure analysis
├── experiments/        # Ablation studies
├── utils/              # Logging & Checkpointing
└── notebooks/          # Analysis & Testing

```

---

## 📝 Execution Roadmap & Task Checklist

### Phase 1: Configuration & Planning (Day 1)

- [ ] **Task 1: `config/model_config.py**`
- Define `ModelConfig` dataclass (vision_dim, text_dim, heads, layers, patch_size).

- [ ] **Task 2: `config/training_config.py**`
- Define `TrainingConfig` (batch_size, learning_rate, epochs, device).

- [ ] **Task 3: `config/dataset_config.py**`
- Define paths for BDD100K images and JSONs, and normalization constants.

### Phase 2: Data Pipeline (Day 2)

- [ ] **Task 4: `data/dataset_builder.py**`
- Implement `process_bdd100k`: Filter clear/daytime images, extract bounding boxes.

- [ ] **Task 5: `data/command_generator.py**`
- Implement `generate_pair`: Create "Question + Answer" pairs based on image metadata.

- [ ] **Task 6: `data/tokenizer.py**`
- Build `SimpleTokenizer`: Create vocab dict, `encode()`, and `decode()` methods.

- [ ] **Task 7: `data/data_loader.py**`
- Implement `TrafficDataset` and `collate_fn` to return batched tensors.

### Phase 3: Vision Encoder (Day 3)

- [ ] **Task 8: `model/vision/siglip_encoder.py**`
- Assemble the full Vision Transformer stack.

- [ ] **Task 9: `model/vision/vision_embeddings.py**`
- Implement Patch Embeddings (Conv2d) + Positional Embeddings.

- [ ] **Task 10: `model/vision/vision_attention.py**`
- Implement Multi-Head Self-Attention for vision patches.

### Phase 4: Language Decoder (Day 4)

- [ ] **Task 11: `model/language/gemma_decoder.py**`
- Build the decoder stack (Linear -> Transformer Blocks -> Output Head).

- [ ] **Task 12: `model/language/decoder_layer.py**`
- Implement `DecoderBlock`: Self-Attention + **Cross-Attention** + FFN.

- [ ] **Task 13: `model/language/rope_embeddings.py**`
- Implement Rotary Positional Embeddings (RoPE) for the query/key vectors.

### Phase 5: Multimodal Fusion (Day 5)

- [ ] **Task 14: `model/fusion/projection_layer.py**`
- Implement Linear Projection to map Vision Dim → Language Dim.

- [ ] **Task 15: `model/fusion/cross_attention.py**`
- Implement Cross-Attention (Query=Text, Key/Value=Vision).

- [ ] **Task 16: `model/fusion/multimodal_fusion.py**`
- Implement helper logic for mask creation and token concatenation.

- [ ] **Task 17: `model/vlm_model.py**`
- **Assemble the VLM:** Initialize Encoder, Projector, and Decoder. Write the `forward()` pass.

### Phase 6: Training Infrastructure (Day 6)

- [ ] **Task 18: `training/trainer.py**`
- Implement the main training loop, validation step, and GPU transfer.

- [ ] **Task 19: `training/loss_functions.py**`
- Implement CrossEntropyLoss (with optional masking/weighting).

- [ ] **Task 20: `training/optimizer.py**`
- Configure AdamW with weight decay.

- [ ] **Task 21: `training/scheduler.py**`
- Implement learning rate warmup + cosine decay.

### Phase 7: Evaluation & Metrics (Day 7)

- [ ] **Task 22: `evaluation/metrics.py**`
- Implement Accuracy, Precision, Recall, F1 Score calculations.

- [ ] **Task 23: `evaluation/attention_metrics.py**`
- Implement Attention Entropy and Grounding Score logic.

- [ ] **Task 24: `evaluation/evaluator.py**`
- Create the inference pipeline to run tests on the holdout set.

### Phase 8: Visualization (Day 8)

- [ ] **Task 25: `visualization/attention_viz.py**`
- Create plots for Vision and Language self-attention.

- [ ] **Task 26: `visualization/cross_attention_viz.py**`
- **Critical:** Overlay attention heatmaps onto original images to show reasoning.

- [ ] **Task 27: `visualization/failure_analysis.py**`
- Script to save and display worst-prediction examples.

### Phase 9: Ablation Studies (Day 9)

- [ ] **Task 28: `experiments/ablation_studies.py**`
- Script to run training with components disabled (e.g., no cross-attention).

- [ ] **Task 29: `experiments/sensitivity_analysis.py**`
- Test model robustness against noise or occlusion.

### Phase 10: Utilities & Finalization (Day 10)

- [ ] **Task 30: `utils/checkpoint_manager.py**`
- Logic for saving/loading model weights safely.

- [ ] **Task 31: `utils/logging_utils.py**`
- Wrappers for TensorBoard or simple CSV logging.

- [ ] **Task 32: `utils/tensor_utils.py**`
- Helper functions for padding sequences and creating masks.

- [ ] **Task 33: `notebooks/data_exploration.ipynb**`
- Initial dataset analysis.

- [ ] **Task 34: `notebooks/model_testing.ipynb**`
- Interactive testing sandbox.

- [ ] **Task 35: `notebooks/results_analysis.ipynb**`
- Final report generation.

---

## 📊 Milestones

- **Milestone 1:** Data Pipeline working (Batch of images/text loads correctly).
- **Milestone 2:** Vision Encoder outputs correct tensor shape.
- **Milestone 3:** Language Decoder generates random text (untrained).
- **Milestone 4:** **Full Model Integration** (Forward pass works without error).
- **Milestone 5:** Training Loop runs and loss decreases.
- **Milestone 6:** Evaluation metrics implemented.
- **Milestone 7:** **Visualizations prove grounding** (Model looks at the right objects).

---

## Usage

**To Train:**

```bash
python -m training.trainer

```

**To Evaluate:**

```bash
python -m evaluation.evaluator

```

**To Visualize Attention:**

```bash
python -m visualization.attention_viz --image "test.jpg" --text "Is the road clear?"

```

---

## 📄 License

This project is for educational and research purposes.
**Author:** [Your Name]
**Date:** December 2025
