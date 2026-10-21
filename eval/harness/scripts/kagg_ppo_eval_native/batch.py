"""Use Rust features with the same greedy decoder as ordinary inference."""
import numpy as np

from scripts.kagg_bc_kit.optimized import BatchAgent


class NativeBatchAgent(BatchAgent):
    def feature_arrays(self, observations):
        from scripts.kagg_bc_kit.contracts import require_mask
        for obs in observations:
            require_mask(obs["_native_pool"],[obs["_native_index"]],self.mask_version)
        return [np.stack([obs['_native_pool'].features[key][obs['_native_index']]
                          for obs in observations])
                for key in ('tiles', 'ou', 'pu', 'g', 'legal')]
