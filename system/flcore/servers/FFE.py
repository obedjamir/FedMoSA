import numpy as np
from typing import Dict, List


class FederatedFingerprintAggregator:
    """
    Federated Fingerprint Extraction (FFE) Aggregator.

    Faithful to FednnU-Net paper:
    - Concatenates raw per-sample shapes
    - Concatenates raw per-sample spacings
    - Recomputes global medians
    - Does NOT average summary statistics
    """

    def __init__(self):
        self.client_fingerprints: List[Dict] = []

    def add_client_fingerprint(self, fingerprint: Dict):
        if "raw_shapes" not in fingerprint or "raw_spacings" not in fingerprint:
            raise ValueError(
                "Fingerprint must contain 'raw_shapes' and 'raw_spacings' "
                "for FFE-faithful aggregation."
            )
        self.client_fingerprints.append(fingerprint)

    def aggregate(self) -> Dict:

        if len(self.client_fingerprints) == 0:
            raise ValueError("No client fingerprints added!")

        all_shapes = []
        all_spacings = []
        total_samples = 0

        # 🔥 TRUE FFE BEHAVIOR: concatenate raw per-sample lists
        for fp in self.client_fingerprints:
            all_shapes.extend(fp["raw_shapes"])
            all_spacings.extend(fp["raw_spacings"])
            total_samples += fp["num_samples"]

        all_shapes = np.array(all_shapes)
        all_spacings = np.array(all_spacings)

        # Recompute global statistics centrally
        global_fingerprint = {
            "spacing": np.median(all_spacings, axis=0).tolist(),
            "shape_after_cropping": np.median(all_shapes, axis=0).tolist(),
            "median_image_size_in_voxels": np.median(all_shapes, axis=0).tolist(),
            "num_channels": 1,  # slice-based assumption
            "num_samples": total_samples,

            # Keep raw lists for possible re-aggregation
            "raw_shapes": all_shapes.tolist(),
            "raw_spacings": all_spacings.tolist(),
        }

        print("✅ Federated fingerprint aggregated (TRUE FFE)")
        print(f"   Total slices: {total_samples}")
        print(f"   Global median shape: {global_fingerprint['shape_after_cropping']}")
        print(f"   Global median spacing: {global_fingerprint['spacing']}")

        return global_fingerprint
