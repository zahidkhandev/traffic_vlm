Here is the detailed documentation for the configuration and training infrastructure of your Traffic VLM project. These configurations define a "Nano" scale Vision-Language Model optimized for real-time traffic safety classification.

### **1. Architecture Configuration (`model_config.py`)**

This file defines the structural hyperparameters of the Vision Transformer (ViT) backbone.

| Parameter        | Value | Reason & Effect                                                                                                                                                                                                                                                                                     |
| ---------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`image_size`** | `224` | **Standardization.** This is the standard resolution for ImageNet pre-training. **Effect:** It balances computational efficiency with sufficient detail to see traffic lights and pedestrians.                                                                                                      |
| **`patch_size`** | `16`  | **Token Granularity.** The image is split into 16x16 pixel squares. **Effect:** patches per side, resulting in visual tokens. This is the standard "ViT-B/16" configuration, providing a good trade-off between sequence length and local detail.                                                   |
| **`hidden_dim`** | `512` | **Model Capacity.** A "Nano" scale dimension (standard ViT-Base is 768). **Effect:** Reduces VRAM usage and speeds up inference significantly, making it suitable for edge devices (like a car dashboard), though it limits the model's ability to learn extremely complex, abstract relationships. |
| **`num_layers`** | `8`   | **Depth.** A shallow network (standard is 12). **Effect:** Faster forward passes. For a specific task like "stop/go," massive depth is often unnecessary.                                                                                                                                           |
| **`num_heads`**  | `8`   | **Attention Width.** Maintains the standard head dimension ratio (). **Effect:** Ensures the model can attend to multiple distinct features (e.g., one head for lane lines, one for traffic lights) simultaneously.                                                                                 |
| **`vocab_size`** | `16`  | **Task Specificity.** Extremely small vocabulary. **Effect:** This confirms the model is a **classifier**, not a chatbot. It only knows traffic-specific tokens, removing the overhead of a 30k+ token tokenizer.                                                                                   |

---

### **2. Dataset & Label Configuration (`dataset_config.py`)**

This file manages data preprocessing and the crucial mapping of real-world concepts to model targets.

- **Normalization (`mean`, `std`)**:
- **Config:** Uses standard ImageNet statistics (`mean=[0.485...]`, `std=[0.229...]`).
- **Why:** Even if training from scratch, these values tend to center RGB data effectively.
- **Effect:** faster convergence and gradients that don't explode/vanish early in training.

- **Label Map (`label_map`)**:
- **Config:** Explicitly maps 5 classes: `safe`, `stop_red_light`, `stop_pedestrian`, `stop_vehicle`, `stop_obstacle`.
- **Why:** You removed the generic "NO" class to force the model to identify the _cause_ of the stop.
- **Effect:** This reduces **modal collapse** where a model might just guess "NO" for everything without understanding why. It improves explainability.

- **Environment Filtering (`allowed_time`)**:
- **Config:** Includes `daytime`, `night`, `dawn/dusk`.
- **Effect:** Ensures the dataset covers the challenging lighting conditions seen in your uploaded images (e.g., the night scene in `image_321747.jpg`).

---

### **3. Training Hyperparameters (`training_config.py`)**

This file controls the training loop dynamics.

- **`batch_size: 64`**:
- **Why:** A relatively large batch size for a VLM.
- **Effect:** Stabilizes gradient estimates. Because your model is "Nano" sized, you can fit 64 images on consumer GPUs, which speeds up training compared to smaller batches.

- **`learning_rate: 3e-4`**:
- **Why:** The "Karpathy Constant." This is a standard, robust starting learning rate for Adam-based optimizers on Transformers.
- **Effect:** High enough to learn quickly, low enough to avoid divergence.

- **`mixed_precision: True`**:
- **Why:** Uses FP16 (16-bit floating point) for calculations but keeps FP32 master weights.
- **Effect:** Reduces memory usage by ~40-50% and speeds up math on Tensor Core GPUs (NVIDIA RTX series), allowing for the larger batch size.

---

### **4. Optimizer Strategy (`optimizer.py`)**

This configuration implements **Decoupled Weight Decay** (AdamW).

- **Parameter Splitting**:
- **Logic:** The code iterates through the model and separates parameters into `decay_params` and `no_decay_params`.
- **Why:** Weight decay (L2 regularization) pushes weights towards zero to prevent overfitting. However, applying this to **Biases** and **LayerNorm** weights is harmful because those parameters control the "center" and "scale" of data, not just feature importance.
- **Effect:** This separation is critical for Transformer stability. It prevents the model from collapsing its normalization layers, leading to significantly better final accuracy.

---

### **5. Learning Rate Schedule (`scheduler.py`)**

- **Cosine with Warmup**:
- **Warmup Phase:** Linearly increases LR from 0 to `3e-4` over `num_warmup_steps`.
- **Why:** Transformer gradients can be massive at initialization. Warmup allows the optimizer to gather reliable statistics before taking big steps.

- **Cosine Decay Phase:** Smoothly decreases LR to 0.
- **Why:** A smooth curve allows the model to settle into sharper/better local minima than a "Step" scheduler (which drops LR abruptly).

---

### **6. Loss Function Strategy (`loss_functions.py`)**

- **Label Smoothing (`0.1`)**:
- **Why:** Instead of forcing the model to predict `[0, 1, 0, 0, 0]`, it targets `[0.025, 0.9, 0.025, 0.025, 0.025]`.
- **Effect:** Prevents the model from becoming overconfident. In traffic scenarios, visual ambiguity is common (e.g., a distant light). Smoothing encourages the model to learn robust features rather than memorizing training outliers.

- **Removal of Class Weights**:
- **Documentation Note:** The file explicitly states: _"We removed class_weights because we are now using WeightedRandomSampler"_.
- **Effect:** This is a crucial correction. If you used both **Sampler** (oversampling rare data) AND **Loss Weights** (penalizing errors on rare data more), you would "double count" the importance of rare classes (like `Stop: Obstacle`), causing the model to hallucinate obstacles everywhere.

### **Summary of System Interactions**

1. **Input:** Images are normalized and filtered for diverse lighting (`dataset_config`).
2. **Model:** A lightweight ViT processes them into 196 tokens (`model_config`).
3. **Training:** The `Trainer` uses Mixed Precision for speed (`trainer.py`), while the `Optimizer` carefully applies regularization only where needed (`optimizer.py`).
4. **Learning:** The `Scheduler` warms up the training to prevent early collapse, and `Label Smoothing` prevents late-stage overfitting (`loss_functions`).
