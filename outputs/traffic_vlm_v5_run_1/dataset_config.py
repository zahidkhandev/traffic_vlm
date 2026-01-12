import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatasetConfig:
    root_dir: str = os.path.join("data", "raw")
    output_dir: str = os.path.join("data", "processed")

    images_dir_name: str = "images"
    labels_dir_name: str = "labels"

    image_size: int = 224
    max_seq_len: int = 32

    # ImageNet Mean/Std
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    allowed_weather: List[str] = field(
        default_factory=lambda: [
            "clear",
            "partly cloudy",
            "overcast",
            "rainy",
            "snowy",
            "foggy",
            "undefined",
        ]
    )
    allowed_time: List[str] = field(
        default_factory=lambda: ["daytime", "night", "dawn/dusk", "undefined"]
    )

    # --- UPDATED LABEL MAP (NO "NO") ---
    # 0 = Safe
    # 1 = Stop Red Light
    # 2 = Stop Pedestrian
    # 3 = Stop Vehicle
    # 4 = Stop Obstacle
    label_map: Dict[str, int] = field(
        default_factory=lambda: {
            "safe": 0,
            "stop_red_light": 1,
            "stop_pedestrian": 2,
            "stop_vehicle": 3,
            "stop_obstacle": 4,
        }
    )

    num_classes: int = 5
