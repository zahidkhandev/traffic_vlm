import json

import numpy as np
from cleanlab.filter import find_label_issues


class CleanLabQC:
    """Layer 3: Statistical QC using Confident Learning"""

    def __init__(self):
        pass

    def find_errors(self, labels, pred_probs):
        """
        Find label errors using Cleanlab

        Args:
            labels: List of integer labels [0, 1, 2, 3, 4]
            pred_probs: numpy array (N, 5) - probabilities from your model

        Returns:
            error_indices: Indices of potentially mislabeled samples
        """
        issues = find_label_issues(
            labels=np.array(labels),
            pred_probs=pred_probs,
            return_indices_ranked_by="self_confidence",
        )

        error_indices = np.where(issues)[0]
        print(f"[CleanLab] Found {len(error_indices)} potential label errors")

        return error_indices
