# Traffic Scene Command Understanding VLM - ULTRA-DETAILED Project Blueprint

## PaliGemma Architecture for BDD100K on Windows/Local Machine

---

## 🎯 Development Environment Specifications

```
Hardware:
- GPU: NVIDIA RTX A3000 (6GB VRAM)
- RAM: 16GB minimum
- Storage: 50GB free space
- OS: Windows 10/11

Software Stack:
- Python 3.10+
- PyTorch 2.0+ with CUDA 11.8
- CUDA Toolkit installed
- cuDNN configured

Dataset Location:
BDD100K_ROOT = "C:/Users/YourName/Downloads/bdd100k"

Project Root:
PROJECT_ROOT = "C:/Users/YourName/Documents/traffic_vlm"
```

---

## 📂 Complete Project Structure (Expanded)

```
C:/Users/YourName/Documents/traffic_vlm/
│
├── config/
│   ├── __init__.py
│   ├── model_config.py              # Task 1
│   ├── training_config.py           # Task 2
│   ├── dataset_config.py            # Task 3
│   └── paths_config.py              # NEW: All file paths
│
├── data/
│   ├── __init__.py
│   ├── raw/                         # Raw BDD100K data
│   │   ├── images/
│   │   │   ├── 100k/
│   │   │   │   ├── train/
│   │   │   │   ├── val/
│   │   │   │   └── test/
│   │   └── labels/
│   │       └── bdd100k_labels_images_train.json
│   │
│   ├── processed/                   # Processed data
│   │   ├── train_images.h5
│   │   ├── val_images.h5
│   │   ├── test_images.h5
│   │   ├── train_commands.json
│   │   ├── val_commands.json
│   │   ├── test_commands.json
│   │   ├── vocab.json
│   │   └── dataset_stats.json
│   │
│   ├── dataset_builder.py           # Task 4
│   ├── command_generator.py         # Task 5
│   ├── tokenizer.py                 # Task 6
│   ├── data_loader.py               # Task 7
│   └── bdd100k_parser.py            # NEW: BDD100K format handler
│
├── model/
│   ├── __init__.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── siglip_encoder.py       # Task 8
│   │   ├── vision_embeddings.py    # Task 9
│   │   ├── vision_attention.py     # Task 10
│   │   └── normalization.py        # NEW: RMS/Layer norm
│   │
│   ├── language/
│   │   ├── __init__.py
│   │   ├── gemma_decoder.py        # Task 11
│   │   ├── decoder_layer.py        # Task 12
│   │   ├── rope_embeddings.py      # Task 13
│   │   ├── text_embeddings.py      # NEW: Word embeddings
│   │   └── attention.py            # NEW: GQA implementation
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── projection_layer.py     # Task 14
│   │   ├── cross_attention.py      # Task 15
│   │   └── multimodal_fusion.py    # Task 16
│   │
│   └── vlm_model.py                # Task 17
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # Task 18
│   ├── loss_functions.py           # Task 19
│   ├── optimizer.py                # Task 20
│   ├── scheduler.py                # Task 21
│   └── mixed_precision.py          # NEW: AMP handler
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                  # Task 22
│   ├── attention_metrics.py        # Task 23
│   ├── evaluator.py                # Task 24
│   └── confusion_matrix.py         # NEW: Detailed confusion analysis
│
├── visualization/
│   ├── __init__.py
│   ├── attention_viz.py            # Task 25
│   ├── cross_attention_viz.py      # Task 26
│   ├── failure_analysis.py         # Task 27
│   └── plot_utils.py               # NEW: Matplotlib helpers
│
├── experiments/
│   ├── __init__.py
│   ├── ablation_studies.py         # Task 28
│   ├── sensitivity_analysis.py     # Task 29
│   └── experiment_runner.py        # NEW: Batch experiment orchestration
│
├── utils/
│   ├── __init__.py
│   ├── checkpoint_manager.py       # Task 30
│   ├── logging_utils.py            # Task 31
│   ├── tensor_utils.py             # Task 32
│   ├── memory_profiler.py          # NEW: GPU memory tracking
│   └── reproducibility.py          # NEW: Seed management
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Task 33
│   ├── 02_bdd100k_analysis.ipynb   # NEW: BDD100K specific analysis
│   ├── 03_model_testing.ipynb      # Task 34
│   ├── 04_attention_debugging.ipynb # NEW: Debug attention issues
│   └── 05_results_analysis.ipynb   # Task 35
│
├── scripts/
│   ├── download_bdd100k.py         # NEW: Dataset download helper
│   ├── preprocess_data.py          # NEW: One-click preprocessing
│   ├── train.py                    # NEW: Main training script
│   ├── evaluate.py                 # NEW: Main evaluation script
│   └── visualize_attention.py      # NEW: Generate attention plots
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_model_forward.py
│   ├── test_attention_shapes.py
│   └── test_training_step.py
│
├── outputs/
│   ├── checkpoints/                # Model checkpoints
│   ├── logs/                       # TensorBoard logs
│   ├── visualizations/             # Generated plots
│   └── results/                    # Evaluation results
│
├── docs/
│   ├── architecture.md             # Architecture decisions
│   ├── bdd100k_setup.md           # BDD100K setup guide
│   ├── training_guide.md           # Training instructions
│   └── attention_analysis.md       # Attention interpretation guide
│
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

---

## 📝 ULTRA-DETAILED Task Breakdown

---

## **PHASE 1: CONFIGURATION & PLANNING (Day 1)**

---

### **Task 1: model_config.py - Define Model Architecture Parameters**

#### **1.1 File Structure**

Create a Python dataclass or dictionary with the following sections:

**1.1.1 Vision Encoder Configuration**

```
VisionConfig:
- image_size: int (128 or 224)
- patch_size: int (16)
- num_channels: int (3 for RGB)
- hidden_size: int (256 or 512)
- num_hidden_layers: int (6 or 12)
- num_attention_heads: int (4 or 8)
- intermediate_size: int (hidden_size * 4)
- hidden_act: str ("gelu")
- layer_norm_eps: float (1e-6)
- attention_dropout: float (0.0 or 0.1)
- initializer_range: float (0.02)
```

**Decision Tree for Vision Config:**

- **If GPU VRAM < 6GB**: image_size=128, hidden_size=256, num_layers=6, heads=4
- **If GPU VRAM = 6GB (your case)**: image_size=224, hidden_size=384, num_layers=8, heads=6
- **If GPU VRAM > 8GB**: image_size=224, hidden_size=512, num_layers=12, heads=8

**Calculate Number of Patches:**

```
num_patches = (image_size // patch_size) ** 2
For 224x224 with 16x16 patches: 196 patches
For 128x128 with 16x16 patches: 64 patches
```

**1.1.2 Language Decoder Configuration**

```
LanguageConfig:
- vocab_size: int (300-500 for traffic domain)
- max_position_embeddings: int (128 for commands)
- hidden_size: int (should match projected vision size)
- num_hidden_layers: int (4-6)
- num_attention_heads: int (4-8)
- num_key_value_heads: int (2-4 for GQA)
- intermediate_size: int (hidden_size * 4)
- hidden_act: str ("gelu" or "swiglu")
- rope_theta: float (10000.0)
- layer_norm_eps: float (1e-6)
- attention_dropout: float (0.1)
- use_cache: bool (True for inference)
```

**GQA Configuration Decision:**

- Standard Multi-Head: num_key_value_heads = num_attention_heads
- GQA 4:1 ratio: num_attention_heads=8, num_key_value_heads=2
- GQA 2:1 ratio: num_attention_heads=8, num_key_value_heads=4

**1.1.3 Projection Layer Configuration**

```
ProjectionConfig:
- vision_hidden_size: int (from vision config)
- language_hidden_size: int (from language config)
- projection_type: str ("linear" or "mlp")
- use_layer_norm: bool (True)
- dropout: float (0.0)
```

**1.1.4 Task-Specific Configuration**

```
TaskConfig:
- task_type: str ("classification" or "generation")
- num_classes: int (2 for YES/NO, 3 for YES/NO/MAYBE)
- output_vocab_size: int (if generation)
- max_generation_length: int (20 tokens)
```

**1.1.5 Model Size Estimation**
Calculate total parameters:

```
Vision Encoder Parameters:
- Patch embeddings: 3 * patch_size^2 * hidden_size
- Position embeddings: num_patches * hidden_size
- Transformer layers: num_layers * (
    - Attention QKV: 3 * hidden_size^2
    - Attention out: hidden_size^2
    - FFN: 2 * hidden_size * intermediate_size
    - Layer norms: 4 * hidden_size per layer
  )

Language Decoder Parameters:
- Word embeddings: vocab_size * hidden_size
- Position embeddings (via RoPE): 0 trainable params
- Transformer layers: Similar calculation
- Output head: hidden_size * num_classes (or vocab_size)

Total Target: < 50M parameters for A3000
```

**1.1.6 Memory Budget Calculation**

```
Batch Size Estimation:
- Model parameters: ~40M * 4 bytes (FP32) = 160MB
- Activations per sample: ~50MB (depends on sequence length)
- Gradient storage: Same as parameters = 160MB
- Optimizer states (AdamW): 2x parameters = 320MB
- Total per batch item: ~230MB
- Available VRAM: 6GB - 1GB (PyTorch overhead) = 5GB
- Max batch size: 5000MB / 230MB ≈ 21
- **Recommended**: batch_size=4 with gradient accumulation=4 (effective=16)
```

**1.1.7 Config File Template Structure**

```
Create nested dictionary or dataclass:
{
  "model_name": "TrafficVLM-v1",
  "vision": { ... },
  "language": { ... },
  "projection": { ... },
  "task": { ... },
  "training": { ... },  # Reference from training_config
  "inference": {
    "use_kv_cache": True,
    "max_new_tokens": 20,
    "temperature": 1.0,
    "top_p": 0.9
  }
}
```

**1.1.8 Validation Checks to Implement**

- [ ] Verify hidden_size is divisible by num_heads
- [ ] Verify image_size is divisible by patch_size
- [ ] Calculate total parameters < 50M
- [ ] Estimate memory footprint < 5GB
- [ ] Check GQA ratio is valid (num_heads % num_kv_heads == 0)

**1.1.9 Config Versioning**

- Save config with timestamp
- Version control: v1.0, v1.1, etc.
- Log all hyperparameters to TensorBoard/W&B
- Save config JSON alongside every checkpoint

**1.1.10 Export Formats**

- Python dict for runtime
- JSON file for serialization
- YAML file for human readability
- Dataclass for type safety

---

### **Task 2: training_config.py - Training Hyperparameters**

#### **2.1 Optimizer Configuration**

**2.1.1 AdamW Settings**

```
OptimizerConfig:
- optimizer_type: "adamw"
- learning_rate: float (1e-4 to 5e-4)
- weight_decay: float (0.01)
- beta1: float (0.9)
- beta2: float (0.999)
- epsilon: float (1e-8)
- amsgrad: bool (False)
```

**Learning Rate Decision Tree:**

```
If training from scratch:
  - Initial LR: 1e-4
  - Peak LR after warmup: 5e-4

If using pretrained vision encoder:
  - Vision encoder LR: 1e-5 (10x smaller)
  - Language decoder LR: 1e-4
  - Projection layer LR: 5e-4 (highest)
```

**2.1.2 Parameter Groups**
Define separate LR for different components:

```
parameter_groups = [
  {
    "params": vision_encoder.parameters(),
    "lr": 1e-5,
    "weight_decay": 0.01,
    "name": "vision_encoder"
  },
  {
    "params": projection_layer.parameters(),
    "lr": 5e-4,
    "weight_decay": 0.0,  # Don't decay projection
    "name": "projection"
  },
  {
    "params": language_decoder.parameters(),
    "lr": 1e-4,
    "weight_decay": 0.01,
    "name": "language_decoder"
  },
  {
    "params": [biases, layer_norms],
    "lr": 1e-4,
    "weight_decay": 0.0,  # Never decay biases/norms
    "name": "no_decay"
  }
]
```

**2.1.3 Learning Rate Schedule**

```
SchedulerConfig:
- scheduler_type: "cosine_with_warmup"
- num_warmup_steps: int (500-1000)
- num_training_steps: int (total_steps)
- num_cycles: float (0.5 for cosine)
- lr_end: float (1e-7)
```

**Warmup Strategy:**

```
Warmup steps calculation:
- Dataset size: 5000 samples
- Batch size: 4
- Gradient accumulation: 4
- Steps per epoch: 5000 / (4 * 4) = 312 steps
- Total epochs: 20
- Total steps: 312 * 20 = 6240
- Warmup: 10% of total = 624 steps

LR Schedule:
Step 0-624: Linear warmup 1e-6 → 1e-4
Step 624-6240: Cosine decay 1e-4 → 1e-7
```

#### **2.2 Training Loop Configuration**

**2.2.1 Batch Configuration**

```
BatchConfig:
- train_batch_size: int (4 for A3000)
- eval_batch_size: int (8, can be higher)
- gradient_accumulation_steps: int (4)
- effective_batch_size: int (16)
- num_workers: int (2 for Windows)
- pin_memory: bool (True)
- prefetch_factor: int (2)
```

**Memory-Batch Size Matrix:**

```
| VRAM  | Batch Size | Gradient Accum | Effective |
|-------|------------|----------------|-----------|
| 6GB   | 4          | 4              | 16        |
| 8GB   | 8          | 2              | 16        |
| 12GB  | 16         | 1              | 16        |
```

**2.2.2 Mixed Precision Configuration**

```
MixedPrecisionConfig:
- enabled: bool (True)
- dtype: str ("float16" or "bfloat16")
- loss_scale: str ("dynamic")
- init_scale: float (2^16)
- growth_interval: int (2000)
```

**FP16 vs BF16 Decision:**

```
Use FP16 if:
- GPU supports it (A3000 does)
- Training is stable
- Need maximum memory savings

Use BF16 if:
- GPU supports it (Ampere+ only)
- Training has numerical instability
- Don't want to tune loss scaling
```

**2.2.3 Gradient Configuration**

```
GradientConfig:
- max_grad_norm: float (1.0)
- gradient_clipping_type: str ("norm" or "value")
- detect_anomaly: bool (False, True for debugging)
```

**2.2.4 Training Duration**

```
TrainingDurationConfig:
- num_epochs: int (20-30)
- max_steps: int (6000-8000)
- early_stopping_patience: int (5 epochs)
- early_stopping_metric: str ("val_accuracy")
- early_stopping_mode: str ("max")
```

#### **2.3 Validation & Checkpointing**

**2.3.1 Validation Schedule**

```
ValidationConfig:
- eval_strategy: str ("steps" or "epoch")
- eval_steps: int (100)
- eval_accumulation_steps: int (None)
- save_strategy: str ("steps")
- save_steps: int (500)
- save_total_limit: int (5)
- load_best_model_at_end: bool (True)
```

**2.3.2 Checkpoint Configuration**

```
CheckpointConfig:
- checkpoint_dir: str ("outputs/checkpoints")
- save_optimizer: bool (True)
- save_scheduler: bool (True)
- save_rng_state: bool (True)
- checkpoint_format: str ("pytorch" or "safetensors")
```

**Checkpoint Naming Convention:**

```
checkpoint-epoch{epoch:02d}-step{step:06d}-acc{acc:.4f}.pt

Example:
checkpoint-epoch05-step1560-acc0.7234.pt
```

**2.3.3 Logging Configuration**

```
LoggingConfig:
- logging_dir: str ("outputs/logs")
- logging_strategy: str ("steps")
- logging_steps: int (10)
- log_level: str ("info")
- report_to: list (["tensorboard", "wandb"])
- project_name: str ("traffic-vlm")
- run_name: str ("experiment-{timestamp}")
```

#### **2.4 Regularization**

**2.4.1 Dropout Configuration**

```
DropoutConfig:
- attention_dropout: float (0.1)
- hidden_dropout: float (0.1)
- embedding_dropout: float (0.1)
- classifier_dropout: float (0.1)
```

**2.4.2 Data Augmentation (for images)**

```
AugmentationConfig:
- random_horizontal_flip: float (0.5)
- color_jitter: dict ({
    "brightness": 0.1,
    "contrast": 0.1,
    "saturation": 0.1,
    "hue": 0.05
  })
- random_rotation: float (5 degrees)
- random_crop: bool (False, don't crop traffic scenes)
- normalize: dict ({
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  })
```

**2.4.3 Label Smoothing**

```
LabelSmoothingConfig:
- enabled: bool (True)
- smoothing: float (0.1)
```

#### **2.5 Reproducibility**

**2.5.1 Random Seed Configuration**

```
SeedConfig:
- manual_seed: int (42)
- numpy_seed: int (42)
- torch_seed: int (42)
- cudnn_deterministic: bool (True)
- cudnn_benchmark: bool (False)
```

**2.5.2 Deterministic Settings**

```
Set the following:
- torch.use_deterministic_algorithms(True)
- os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
- Worker init function with unique seeds
```

#### **2.6 Hardware-Specific Configuration**

**2.6.1 CUDA Settings**

```
CUDAConfig:
- device: str ("cuda" if torch.cuda.is_available() else "cpu")
- device_count: int (1)
- multi_gpu: bool (False)
- cuda_launch_blocking: bool (False, True for debugging)
```

**2.6.2 Windows-Specific Settings**

```
WindowsConfig:
- num_workers: int (2, not 4+ due to Windows multiprocessing)
- persistent_workers: bool (True)
- use_multiprocessing_context: str ("spawn")
```

#### **2.7 Training Phases**

**2.7.1 Multi-Stage Training Strategy**

```
Phase 1: Freeze vision encoder (epochs 1-5)
- Train only projection + language decoder
- LR: 5e-4
- Batch size: 4

Phase 2: Unfreeze all (epochs 6-20)
- Train entire model
- LR: 1e-4 (vision), 1e-4 (language)
- Batch size: 4

Phase 3: Fine-tuning (epochs 21-30)
- Lower LR: 1e-5
- Batch size: 2 (for stability)
```

#### **2.8 Performance Monitoring**

**2.8.1 Metrics to Track**

```
TrainingMetrics:
- loss (per step)
- accuracy (per epoch)
- learning_rate (per step)
- grad_norm (per step)
- gpu_memory_allocated (per step)
- samples_per_second (per step)
- time_per_epoch (per epoch)
```

**2.8.2 Alerts & Monitoring**

```
AlertConfig:
- nan_loss_alert: bool (True)
- grad_norm_threshold: float (100.0)
- memory_threshold: float (5.5GB)
```

#### **2.9 Complete Training Config Template**

```
training_config = {
  "optimizer": { ... },
  "scheduler": { ... },
  "batch": { ... },
  "mixed_precision": { ... },
  "gradient": { ... },
  "duration": { ... },
  "validation": { ... },
  "checkpoint": { ... },
  "logging": { ... },
  "regularization": { ... },
  "reproducibility": { ... },
  "hardware": { ... },
  "phases": [ ... ]
}
```

---

### **Task 3: dataset_config.py - Data Configuration**

#### **3.1 BDD100K Directory Structure Understanding**

**3.1.1 Expected BDD100K Structure**

```
C:/Users/YourName/Downloads/bdd100k/
├── images/
│   └── 100k/
│       ├── train/          # 70,000 images
│       ├── val/            # 10,000 images
│       └── test/           # 20,000 images (no labels)
│
└── labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

**3.1.2 BDD100K Label Format Understanding**

```
JSON Structure:
[
  {
    "name": "b1c66a42-6f7d68ca.jpg",
    "attributes": {
      "weather": "clear",
      "scene": "city street",
      "timeofday": "daytime"
    },
    "labels": [
      {
        "category": "traffic light",
        "attributes": {
          "trafficLightColor": "red"
        },
        "box2d": {
          "x1": 1000.0,
          "y1": 100.0,
          "x2": 1050.0,
          "y2": 200.0
        }
      },
      {
        "category": "pedestrian",
        "box2d": { ... }
      }
    ]
  },
  ...
]
```

**3.1.3 Relevant Categories for Traffic VLM**

```
Primary Categories to Extract:
- "traffic light" (with color: red, green, yellow)
- "traffic sign" (with signType)
- "pedestrian"
- "rider"
- "car"
- "truck"
- "bus"
- "train"
- "motorcycle"
- "bicycle"

Lane Categories:
- "lane" (with direction, style)

Ignore Categories:
- "drivable area"
- "other vehicle"
- "other person"
- "trailer"
```

#### **3.2 Filtering Configuration**

**3.2.1 Scene Attribute Filters**

```
FilterConfig:
  weather:
    - include: ["clear", "partly cloudy"]
    - exclude: ["rainy", "snowy", "foggy"]

  timeofday:
    - include: ["daytime"]
    - exclude: ["night", "dawn/dusk", "undefined"]

  scene:
    - include: ["city street", "highway", "residential"]
    - exclude: ["parking lot", "gas stations", "tunnel"]
```

**Filtering Rationale:**

- Clear weather: Better visibility, easier learning
- Daytime: Consistent lighting
- Urban scenes: More traffic objects

**3.2.2 Quality Filters**

```
QualityConfig:
  image:
    - min_resolution: (720, 1280)  # BDD100K standard
    - max_file_size: 5MB
    - check_corruption: True

  annotations:
    - min_objects_per_image: 1
    - max_objects_per_image: 50
    - min_box_area: 400 pixels (20x20)
    - max_box_area: 90% of image
```

**3.2.3 Object-Based Filters**

```
ObjectFilterConfig:
  required_for_safety_commands:
    - at_least_one_of: ["pedestrian", "traffic light", "car"]

  required_for_detection_commands:
    - at_least_one_of: ["traffic sign", "traffic light"]

  minimum_diversity:
    - unique_categories_per_split: 8
```

#### **3.3 Dataset Split Configuration**

**3.3.1 Split Strategy**

```
SplitConfig:
  source: "bdd100k/images/100k/train"  # Use only train split

  train:
    - percentage: 0.80
    - target_size: 4000 images

  val:
    - percentage: 0.10
    - target_size: 500 images

  test:
    - percentage: 0.10
    - target_size: 500 images

  split_method: "stratified"  # Ensure balanced scene types
  random_seed: 42
```

**3.3.2 Stratification Strategy**

```
Stratify by:
1. Scene type (city/highway/residential)
2. Object count bins (1-5, 6-10, 11-15, 16+)
3. Presence of key objects (traffic light yes/no)

Ensures:
- Each split has similar distribution
- No data leakage
- Balanced difficulty
```

#### **3.4 Image Preprocessing Configuration**

**3.4.1 Resize Strategy**

```
ResizeConfig:
  target_size: (224, 224)  # or (128, 128)
  method: "bilinear"
  maintain_aspect_ratio: False

  padding:
    - if_maintain_aspect: True
    - pad_value: 0
    - pad_mode: "constant"
```

**3.4.2 Normalization**

```
NormalizationConfig:
  method: "imagenet"  # or "dataset_specific"

  imagenet_stats:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

  compute_dataset_stats: True
  save_stats_path: "data/processed/dataset_stats.json"
```

**3.4.3 Data Storage Format**

```
StorageConfig:
  format: "hdf5"  # or "lmdb", "tfrecord"
  compression: "gzip"
  compression_level: 4

  hdf5_structure:
    /train/images: [N, H, W, C] uint8
    /train/metadata: JSON strings
    /val/images: [M, H, W, C] uint8
    /val/metadata: JSON strings
```

#### **3.5 Command Generation Configuration**

**3.5.1 Command Template Categories**

```
CommandCategories:
  1. Safety Commands (40%):
     - "Can the car safely turn left?"
     - "Is it safe to proceed?"
     - "Can the car change lanes?"
     - "Is overtaking allowed?"

  2. Detection Commands (30%):
     - "Is there a pedestrian in the scene?"
     - "Do you see a traffic light?"
     - "Is there a stop sign?"
     - "Are there any vehicles nearby?"

  3. State Commands (20%):
     - "What color is the traffic light?"
     - "Is the traffic light red?"
     - "Is the pedestrian crossing?"

  4. Navigation Commands (10%):
     - "Is the left lane clear?"
     - "Is there an obstacle ahead?"
     - "Can the car turn right?"
```

**3.5.2 Command Complexity Levels**

```
ComplexityConfig:
  simple (60%):
    - Single object query
    - Binary answer
    - Example: "Is there a car?"

  medium (30%):
    - Object + attribute query
    - Binary or categorical answer
    - Example: "Is the traffic light red?"

  complex (10%):
    - Multi-object reasoning
    - Conditional logic
    - Example: "Can the car turn left if there's no pedestrian?"
```

**3.5.3 Answer Distribution**

```
AnswerDistributionConfig:
  binary_tasks:
    YES: 50%
    NO: 50%

  categorical_tasks:
    RED: 33%
    GREEN: 33%
    YELLOW: 33%

  balancing_strategy: "oversample_minority"
```

#### **3.6 Vocabulary Configuration**

**3.6.1 Special Tokens**

```
SpecialTokensConfig:
  pad_token: "[PAD]"
  sos_token: "[SOS]"
  eos_token: "[EOS]"
  unk_token: "[UNK]"
  yes_token: "[YES]"
  no_token: "[NO]"
  maybe_token: "[MAYBE]"
```

**3.6.2 Domain-Specific Vocabulary**

```
VocabConfig:
  traffic_objects:
    ["car", "pedestrian", "traffic", "light", "sign",
     "bus", "truck", "motorcycle", "bicycle", "rider"]

  actions:
    ["turn", "left", "right", "proceed", "stop", "go",
     "overtake", "change", "lane", "crossing"]

  attributes:
    ["red", "green", "yellow", "clear", "safe", "ahead",
     "nearby", "there", "any", "allowed"]

  sentence_structure:
    ["is", "can", "the", "a", "in", "of", "to", "?"]

  max_vocab_size: 500
  min_word_frequency: 2
```

#### **3.7 Data Augmentation Configuration**

**3.7.1 Image Augmentation**

```
ImageAugmentationConfig:
  train_only: True

  spatial:
    horizontal_flip: 0.5
    rotation: (-5, 5)  # degrees
    scale: (0.95, 1.05)

  color:
    brightness: 0.1
    contrast: 0.1
    saturation: 0.1
    hue: 0.05

  noise:
    gaussian_noise: 0.01
    probability: 0.1
```

**3.7.2 Text Augmentation**

```
TextAugmentationConfig:
  enabled: True

  paraphrasing:
    "Is there a car?" → "Do you see a car?"
    "Can the car turn left?" → "Is left turn safe?"

  word_substitution:
    "car" ↔ "vehicle"
    "pedestrian" ↔ "person"

  probability: 0.2
```

#### **3.8 Dataset Statistics to Compute**

**3.8.1 Image Statistics**

```
ImageStats:
  - mean_rgb: [R, G, B]
  - std_rgb: [R, G, B]
  - min_max_values: [[min_R, max_R], ...]
  - aspect_ratio_distribution: histogram
  - brightness_distribution: histogram
```

**3.8.2 Annotation Statistics**

```
AnnotationStats:
  - objects_per_image: {mean, std, min, max, distribution}
  - category_distribution: {category: count}
  - box_size_distribution: histogram
  - scene_type_distribution: {scene: count}
  - weather_distribution: {weather: count}
```

**3.8.3 Command Statistics**

```
CommandStats:
  - total_commands: int
  - commands_per_image: {mean, std}
  - command_length_distribution: histogram
  - answer_distribution: {YES: count, NO: count}
  - command_type_distribution: {safety: count, detection: count, ...}
  - vocabulary_size: int
  - avg_words_per_command: float
```

#### **3.9 Path Configuration**

**3.9.1 Absolute Paths**

```
PathsConfig:
  # Raw data
  bdd100k_root: "C:/Users/YourName/Downloads/bdd100k"
  images_dir: "{bdd100k_root}/images/100k"
  labels_dir: "{bdd100k_root}/labels"

  # Processed data
  processed_root: "C:/Users/YourName/Documents/traffic_vlm/data/processed"
  train_images_h5: "{processed_root}/train_images.h5"
  val_images_h5: "{processed_root}/val_images.h5"
  test_images_h5: "{processed_root}/test_images.h5"

  # Commands
  train_commands_json: "{processed_root}/train_commands.json"
  val_commands_json: "{processed_root}/val_commands.json"
  test_commands_json: "{processed_root}/test_commands.json"

  # Metadata
  vocab_json: "{processed_root}/vocab.json"
  dataset_stats_json: "{processed_root}/dataset_stats.json"
  split_info_json: "{processed_root}/split_info.json"
```

**3.9.2 Relative Paths (for portability)**

```
RelativePathsConfig:
  project_root: get_project_root()
  data_dir: "{project_root}/data"
  raw_dir: "{data_dir}/raw"
  processed_dir: "{data_dir}/processed"
```

#### **3.10 Validation & Testing**

**3.10.1 Data Quality Checks**

```
QualityChecks:
  - [ ] All images loadable
  - [ ] All images have annotations
  - [ ] No duplicate images
  - [ ] Consistent image dimensions
  - [ ] Valid bounding boxes (x1<x2, y1<y2)
  - [ ] Label JSON parseable
  - [ ] Split files non-overlapping
```

**3.10.2 Dataset Sanity Tests**

```
SanityTests:
  - [ ] Train/val/test split percentages correct
  - [ ] Answer distribution balanced
  - [ ] No empty images (min 1 object)
  - [ ] Command vocabulary complete
  - [ ] All commands have answers
  - [ ] Image-command pairs aligned
```

#### **3.11 Complete Dataset Config Template**

```
dataset_config = {
  "bdd100k": {
    "root": "C:/Users/YourName/Downloads/bdd100k",
    "structure": { ... }
  },
  "filtering": { ... },
  "splits": { ... },
  "preprocessing": { ... },
  "commands": { ... },
  "vocabulary": { ... },
  "augmentation": { ... },
  "storage": { ... },
  "paths": { ... },
  "statistics": { ... }
}
```

---

## **PHASE 2: DATA PIPELINE (Day 2)**

---

### **Task 4: dataset_builder.py - BDD100K Processing**

#### **4.1 BDD100K Parser Implementation**

**4.1.1 JSON Label Parser**

```
Class: BDD100KLabelParser
Purpose: Parse raw BDD100K JSON annotations

Methods to implement:
1. load_labels(json_path: str) -> List[dict]
   - Load JSON file
   - Validate structure
   - Return list of image annotations

2. parse_single_annotation(anno: dict) -> ImageAnnotation
   - Extract image name
   - Extract attributes (weather, scene, time)
   - Extract all object labels
   - Return structured object

3. filter_by_attributes(annotations, filters) -> List
   - Apply weather filter
   - Apply time filter
   - Apply scene filter
   - Return filtered list

4. extract_objects(anno: dict) -> List[Object]
   - Parse box2d coordinates
   - Parse category
   - Parse attributes (traffic light color, etc.)
   - Validate box coordinates
   - Return list of objects

5. validate_annotation(anno: dict) -> bool
   - Check required fields present
   - Check box coordinates valid
   - Check category in allowed list
   - Return validation result
```

**4.1.2 Data Structures to Define**

```
ImageAnnotation:
  - image_id: str
  - image_path: str
  - width: int (1280)
  - height: int (720)
  - weather: str
  - scene: str
  - timeofday: str
  - objects: List[Object]

Object:
  - category: str
  - bbox: BoundingBox (x1, y1, x2, y2)
  - attributes: dict
  - area: float
  - is_occluded: bool
  - is_truncated: bool

BoundingBox:
  - x1, y1, x2, y2: float
  - width, height: float
  - center_x, center_y: float
  - area: float
```

**4.1.3 Filtering Logic**

```
Function: apply_filters(annotations, config)

Step 1: Weather filtering
  for anno in annotations:
    if anno.weather not in config.allowed_weather:
      skip anno

Step 2: Time filtering
  for anno in annotations:
    if anno.timeofday != "daytime":
      skip anno

Step 3: Scene filtering
  for anno in annotations:
    if anno.scene in config.excluded_scenes:
      skip anno

Step 4: Object count filtering
  for anno in annotations:
    if len(anno.objects) < config.min_objects:
      skip anno
    if len(anno.objects) > config.max_objects:
      skip anno

Step 5: Object category filtering
  for anno in annotations:
    relevant_objects = [obj for obj in anno.objects
                        if obj.category in config.relevant_categories]
    if len(relevant_objects) == 0:
      skip anno
    anno.objects = relevant_objects

Return: filtered_annotations
```

#### **4.2 Image Processing Pipeline**

**4.2.1 Image Loader**

```
Function: load_and_validate_image(image_path: str)

Steps:
1. Check file exists
2. Open with PIL or cv2
3. Verify dimensions (720x1280)
4. Convert to RGB (if not already)
5. Check for corruption:
   - Try to read all pixels
   - Check for NaN values
6. Return: image_array [H, W, C]

Error handling:
- FileNotFoundError → log and skip
- Image corrupted → log and skip
- Invalid format → log and skip
```

**4.2.2 Image Resizer**

```
Function: resize_image(image: np.ndarray, target_size: tuple)

Options:
A. Simple resize (distorts aspect ratio):
   - cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

B. Aspect-preserving resize + padding:
   1. Compute scaling factor: min(target_w/w, target_h/h)
   2. Resize to (new_w, new_h)
   3. Pad to target_size with zeros
   4. Return padded image + padding info

For traffic scenes: Use option A (no padding)
Reason: Traffic scenes are typically wide, padding wastes patches
```

**4.2.3 Normalization**

```
Function: normalize_image(image: np.ndarray, stats: dict)

Steps:
1. Convert to float32: image = image.astype(np.float32) / 255.0
2. Normalize per channel:
   image[..., 0] = (image[..., 0] - mean[0]) / std[0]
   image[..., 1] = (image[..., 1] - mean[1]) / std[1]
   image[..., 2] = (image[..., 2] - mean[2]) / std[2]
3. Return normalized image

Save normalization stats:
- mean: [R_mean, G_mean, B_mean]
- std: [R_std, G_std, B_std]
```

**4.2.4 Bounding Box Transformation**

```
Function: transform_boxes(boxes, original_size, target_size)

When resizing image, must resize boxes too:

scale_x = target_size[0] / original_size[0]
scale_y = target_size[1] / original_size[1]

for box in boxes:
  box.x1 *= scale_x
  box.x2 *= scale_x
  box.y1 *= scale_y
  box.y2 *= scale_y

Validate transformed boxes:
- x1, y1 >= 0
- x2 <= target_width
- y2 <= target_height
- x2 > x1 and y2 > y1

Return: transformed_boxes
```

#### **4.3 Dataset Statistics Computation**

**4.3.1 Image Statistics**

```
Function: compute_image_statistics(image_paths: List[str])

Compute per-channel mean and std:

Method 1 (Memory efficient):
  running_mean = [0, 0, 0]
  running_std = [0, 0, 0]
  count = 0

  for image_path in image_paths:
    image = load_image(image_path)  # [H, W, 3]
    running_mean += image.mean(axis=(0,1))
    count += 1

  mean = running_mean / count

  for image_path in image_paths:
    image = load_image(image_path)
    running_std += ((image - mean) ** 2).mean(axis=(0,1))

  std = sqrt(running_std / count)

Save to: data/processed/dataset_stats.json
```

**4.3.2 Annotation Statistics**

```
Function: compute_annotation_statistics(annotations)

Collect:
1. Object count per image:
   - histogram: {1: count, 2: count, ..., 20+: count}
   - mean, std, min, max

2. Category distribution:
   - {category: count} for all categories
   - Plot bar chart

3. Bounding box sizes:
   - histogram of areas
   - histogram of aspect ratios
   - mean box size

4. Scene complexity:
   - Define complexity = num_objects + num_categories
   - histogram of complexity scores

5. Attribute distributions:
   - Weather: {clear: count, rainy: count, ...}
   - Time: {daytime: count, night: count, ...}
   - Scene: {city: count, highway: count, ...}

Save to: data/processed/annotation_stats.json
```

#### **4.4 Train/Val/Test Splitting**

**4.4.1 Stratified Splitting Strategy**

```
Function: create_stratified_split(annotations, config)

Stratification keys:
1. Scene type (city/highway/residential)
2. Object count bins (1-5, 6-10, 11-15, 16+)
3. Has traffic light (yes/no)

Steps:
1. Create stratification labels:
   for anno in annotations:
     key = f"{anno.scene}_{get_bin(len(anno.objects))}_{has_traffic_light(anno)}"
     anno.strat_key = key

2. Use sklearn StratifiedShuffleSplit:
   from sklearn.model_selection import StratifiedShuffleSplit
   splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
   train_idx, temp_idx = splitter.split(annotations, strat_keys)

   # Split temp into val and test
   splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
   val_idx, test_idx = splitter2.split(temp_annotations, temp_strat_keys)

3. Verify splits:
   - No overlap between train/val/test
   - Similar distributions across splits
   - Target sizes met (4000/500/500)

4. Save split info:
   {
     "train": [image_id_1, image_id_2, ...],
     "val": [image_id_100, ...],
     "test": [image_id_200, ...]
   }
   Save to: data/processed/split_info.json
```

**4.4.2 Split Validation**

```
Function: validate_split(train, val, test)

Checks:
1. No overlap:
   assert len(set(train) & set(val)) == 0
   assert len(set(train) & set(test)) == 0
   assert len(set(val) & set(test)) == 0

2. Size correct:
   assert len(train) == 4000
   assert len(val) == 500
   assert len(test) == 500

3. Distribution similar:
   for split in [train, val, test]:
     compute scene_distribution
     compute object_count_distribution

   Compare distributions:
   - Chi-square test for categorical (scene type)
   - KS test for numerical (object count)

   Assert p-value > 0.05 (distributions similar)
```

#### **4.5 HDF5 Storage**

**4.5.1 HDF5 File Structure Design**

```
Structure:
/train/
  /images       # [N, 224, 224, 3] uint8
  /image_ids    # [N] strings
  /annotations  # [N] JSON strings

/val/
  /images       # [M, 224, 224, 3] uint8
  /image_ids    # [M] strings
  /annotations  # [M] JSON strings

/test/
  /images       # [K, 224, 224, 3] uint8
  /image_ids    # [K] strings
  /annotations  # [K] JSON strings

/metadata/
  /mean         # [3] float32
  /std          # [3] float32
  /num_train    # scalar int
  /num_val      # scalar int
  /num_test     # scalar int
```

**4.5.2 Writing HDF5**

```
Function: write_to_hdf5(split_data, output_path)

import h5py

with h5py.File(output_path, 'w') as f:
  # Create datasets
  images_dset = f.create_dataset(
    'images',
    shape=(num_images, 224, 224, 3),
    dtype='uint8',
    compression='gzip',
    compression_opts=4
  )

  ids_dset = f.create_dataset(
    'image_ids',
    shape=(num_images,),
    dtype=h5py.string_dtype()
  )

  annos_dset = f.create_dataset(
    'annotations',
    shape=(num_images,),
    dtype=h5py.string_dtype()
  )

  # Write in chunks (to manage memory)
  chunk_size = 100
  for i in range(0, num_images, chunk_size):
    chunk = split_data[i:i+chunk_size]

    # Load and process images
    processed_images = []
    for item in chunk:
      img = load_image(item.image_path)
      img = resize_image(img, (224, 224))
      processed_images.append(img)

    # Write to HDF5
    images_dset[i:i+len(chunk)] = processed_images
    ids_dset[i:i+len(chunk)] = [item.image_id for item in chunk]
    annos_dset[i:i+len(chunk)] = [json.dumps(item.annotation) for item in chunk]

  # Write metadata
  f.create_dataset('metadata/mean', data=mean_rgb)
  f.create_dataset('metadata/std', data=std_rgb)
```

**4.5.3 Reading from HDF5**

```
Function: read_from_hdf5(hdf5_path, index)

with h5py.File(hdf5_path, 'r') as f:
  image = f['images'][index]  # [224, 224, 3]
  image_id = f['image_ids'][index]
  anno_json = f['annotations'][index]
  anno = json.loads(anno_json)

return image, image_id, anno

For batched reading:
  indices = [0, 5, 10, 15]
  images = f['images'][indices]  # HDF5 fancy indexing
```

#### **4.6 Metadata Generation**

**4.6.1 Scene Metadata**

```
For each image, compute:
{
  "image_id": "b1c66a42-6f7d68ca",
  "split": "train",
  "original_size": [720, 1280],
  "processed_size": [224, 224],
  "num_objects": 12,
  "object_categories": ["car", "pedestrian", "traffic light"],
  "scene_type": "city street",
  "weather": "clear",
  "timeofday": "daytime",
  "complexity_score": 3.5,
  "has_traffic_light": true,
  "has_pedestrian": true,
  "num_vehicles": 8
}

Save to: data/processed/train_metadata.json (one per split)
```

**4.6.2 Dataset-Level Metadata**

```
{
  "dataset_name": "BDD100K-Traffic-VLM",
  "version": "1.0",
  "creation_date": "2025-12-17",
  "total_images": 5000,
  "splits": {
    "train": 4000,
    "val": 500,
    "test": 500
  },
  "image_size": [224, 224],
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "object_categories": [...],
  "num_object_categories": 10,
  "scene_types": ["city street", "highway", "residential"],
  "filters_applied": {
    "weather": ["clear", "partly cloudy"],
    "timeofday": ["daytime"]
  }
}

Save to: data/processed/dataset_info.json
```

#### **4.7 Processing Pipeline Orchestration**

**4.7.1 Main Processing Function**

```
Function: build_dataset(config)

Steps:
1. Load BDD100K labels
   - train_labels = load_labels("bdd100k/labels/bdd100k_labels_images_train.json")
   - val_labels = load_labels("bdd100k/labels/bdd100k_labels_images_val.json")

2. Apply filters
   - filtered_train = apply_filters(train_labels, config.filters)
   - filtered_val = apply_filters(val_labels, config.filters)

3. Combine and sample
   - all_filtered = filtered_train + filtered_val
   - sample 5000 images (if more available)

4. Create stratified split
   - train, val, test = create_split(all_filtered, config.split_ratios)

5. Compute statistics
   - stats = compute_statistics(train)
   - save_stats(stats, "data/processed/dataset_stats.json")

6. Process and save images
   - process_and_save_hdf5(train, "data/processed/train_images.h5", stats)
   - process_and_save_hdf5(val, "data/processed/val_images.h5", stats)
   - process_and_save_hdf5(test, "data/processed/test_images.h5", stats)

7. Generate metadata
   - save_metadata(train, "data/processed/train_metadata.json")
   - save_metadata(val, "data/processed/val_metadata.json")
   - save_metadata(test, "data/processed/test_metadata.json")

8. Generate dataset info
   - save_dataset_info(config, stats, "data/processed/dataset_info.json")

9. Validate outputs
   - validate_all_files_exist()
   - validate_hdf5_integrity()
   - validate_splits_non_overlapping()
```

**4.7.2 Progress Tracking**

```
Use tqdm for progress bars:

from tqdm import tqdm

for split in ['train', 'val', 'test']:
  print(f"Processing {split} split...")
  for i in tqdm(range(len(split_data)), desc=f"{split} images"):
    process_image(split_data[i])

Log to file:
logging.info(f"Filtered {len(filtered)} images")
logging.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
logging.info(f"Mean RGB: {mean}")
logging.info(f"Std RGB: {std}")
```

#### **4.8 Error Handling**

**4.8.1 Graceful Failures**

```
Try-except blocks:

failed_images = []

for image_path in image_paths:
  try:
    image = load_image(image_path)
    process_image(image)
  except Exception as e:
    logging.error(f"Failed to process {image_path}: {e}")
    failed_images.append(image_path)
    continue

After processing:
logging.info(f"Successfully processed: {len(image_paths) - len(failed_images)}")
logging.info(f"Failed: {len(failed_images)}")
save_failed_list(failed_images, "data/processed/failed_images.txt")
```

#### **4.9 Testing & Validation**

**4.9.1 Unit Tests**

```
Tests to write:
1. test_load_labels():
   - Load sample JSON
   - Assert correct number of annotations
   - Assert required fields present

2. test_filter_by_weather():
   - Create sample annotations with different weather
   - Apply filter
   - Assert only allowed weather remains

3. test_resize_image():
   - Create 720x1280 image
   - Resize to 224x224
   - Assert shape correct

4. test_hdf5_write_read():
   - Write sample data to HDF5
   - Read back
   - Assert data matches

5. test_stratified_split():
   - Create sample annotations
   - Split
   - Assert no overlap
   - Assert distributions similar
```

**4.9.2 Integration Tests**

```
Test full pipeline:
1. Run build_dataset() on small subset (100 images)
2. Assert all output files created
3. Assert HDF5 files readable
4. Assert metadata JSON parseable
5. Assert statistics reasonable
```
