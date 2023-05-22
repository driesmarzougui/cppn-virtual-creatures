from typing import Dict

import numpy as np
import mrpt


class Archive:
    """
    Distance-based archive as used in Novelty Search experiments.
    MRPT backbone: https://github.com/vioshyvo/mrpt
    """
    def __init__(self, ns_params, archive_restoration=None):
        self.ns_params = ns_params
        self.threshold_params = self.ns_params["archive_threshold"]
        self.k = self.ns_params["k"]
        if archive_restoration is None:
            self.stagnation_step = 0
            self.added = 0

            self.data = [[self.ns_params["behavior_dummy"]] * self.ns_params[
                "behavior_size"]] * 101  # MRPT requires at least a sample size of 101 to autotune

            self.threshold = self.threshold_params["default"]
        else:
            self.stagnation_step = archive_restoration["stagnation_step"]
            self.added = archive_restoration["added"]
            self.data = archive_restoration["data"]
            self.threshold = archive_restoration["threshold"]

        self.archive = self.build_archive(self.data, ns_params)

    def get_restoration(self) -> Dict:
        return {
            "stagnation_step": self.stagnation_step,
            "added": self.added,
            "data": self.data,
            "threshold": self.threshold
        }

    def update(self, data, num_added: int, num_evals: int):
        self.stagnation_step += num_evals
        if num_added > 0:
            self.data = data
            self.archive = self.build_archive(data, self.ns_params)
            self.added += num_added

        if self.threshold_params["mode"] == "dynamic":
            # Check if threshold needs to be updated
            if self.stagnation_step >= self.threshold_params["stagnation"]:
                if self.added == 0:
                    # Reduce threshold
                    self.threshold *= 1 - self.threshold_params["stagnation_reduction"]
                elif self.added > self.threshold_params["stagnation_promotion_threshold"]:
                    # Increase threshold
                    self.threshold *= 1 + self.threshold_params["stagnation_promotion"]

                if self.threshold < self.threshold_params["lower_bound"]:
                    self.threshold = self.threshold_params["lower_bound"]

                self.stagnation_step = 0
                self.added = 0

    @staticmethod
    def build_archive(data, ns_params):
        data = np.array(data).astype(np.float32)
        if len(data) < 101:
            data = np.concatenate(
                (data, [[ns_params["behavior_dummy"]] * ns_params["behavior_size"]] * (101 - len(data)))).astype(
                np.float32)
        archive = mrpt.MRPTIndex(data)
        archive.build_autotune_sample(target_recall=0.9, k=ns_params["k"])
        return archive

    @staticmethod
    def query(archive, behaviors):
        b, d = archive.ann(np.array(behaviors).astype(np.float32), return_distances=True)
        return d, b
