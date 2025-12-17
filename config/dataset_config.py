import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class DatasetConfig:
    root_dir: str = os.path.join("data", "raw")
    output_dir: str = os.path.join("data", "processed")

    images_dir_name: str = "images"
    labels_dir_name: str = "labels"

    image_size: int = 128
    max_seq_len: int = 32

    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    allowed_weather: List[str] = field(
        default_factory=lambda: ["clear", "partly cloudy", "overcast", "undefined"]
    )
    allowed_time: List[str] = field(default_factory=lambda: ["daytime"])
