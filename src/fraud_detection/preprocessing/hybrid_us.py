# =============================================================================
# preprocessing/hybrid_us.py
#
# HybridUS  -  paper Section 2.3, Subsystem 2
#
# Distribution-preserving undersampling method for the NORMAL class.
# Used in the sequential (LSTM) subsystem.
#
# Motivation: oversampling would place synthetic fraud at unknown sequence
# positions, corrupting the temporal structure the LSTM relies on. So
# Subsystem 2 undersamples the normal class instead.
#
# Algorithm (Fig. 2 - right column):
#   Phase 1  OCSVM on normal class -> "normal core" (+1) vs "poorly
#            distributed" (-1). Poorly-distributed normals must NOT be lost.
#   Phase 2  RandomUnderSampler on normal core only.
#   Combine  fraud + undersampled_core + poorly_distributed (order preserved).
#
# Target: fraud = TARGET_FRAUD_RATIO (5%) of the combined training set.
# =============================================================================

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import OneClassSVM

from fraud_detection import config


class HybridUS:
    """HybridUS: distribution-preserving undersampling (paper Section 2.3)."""

    def __init__(
        self,
        target_fraud_ratio=None,
        ocsvm_kernel=None,
        ocsvm_nu=None,
        random_state=None,
    ):
        self.target_fraud_ratio = (
            target_fraud_ratio
            if target_fraud_ratio is not None
            else config.TARGET_FRAUD_RATIO
        )
        self.ocsvm_kernel = ocsvm_kernel if ocsvm_kernel else config.OCSVM_KERNEL
        self.ocsvm_nu = ocsvm_nu if ocsvm_nu is not None else config.OCSVM_NU
        self.random_state = random_state if random_state is not None else config.RANDOM_STATE

    def fit_resample(self, X_train, y_train):
        """Apply HybridUS to the training fold. Temporal order is preserved."""
        X = np.asarray(X_train)
        y = np.asarray(y_train)

        fraud_mask = y == config.FRAUD_LABEL
        normal_mask = y == config.NORMAL_LABEL

        # Record original indices to reconstruct temporal order afterwards
        original_indices = np.arange(len(y))
        fraud_idx = original_indices[fraud_mask]
        normal_idx = original_indices[normal_mask]

        X_fraud = X[fraud_mask]
        X_normal = X[normal_mask]

        n_fraud = len(X_fraud)
        n_normal = len(X_normal)

        print(f"[HybridUS] Input - normal: {n_normal}, fraud: {n_fraud}")

        # -- Phase 1: OCSVM on normal class -----------------------------------
        ocsvm = OneClassSVM(kernel=self.ocsvm_kernel, nu=self.ocsvm_nu)
        ocsvm.fit(X_normal)
        normal_ocsvm_labels = ocsvm.predict(X_normal)  # +1 or -1

        core_mask = normal_ocsvm_labels == 1
        poor_mask = normal_ocsvm_labels == -1

        core_local_idx = np.where(core_mask)[0]  # indices within X_normal
        poor_local_idx = np.where(poor_mask)[0]

        n_core = int(core_mask.sum())
        n_poor = int(poor_mask.sum())

        # Map back to original dataset indices (preserves temporal position)
        poor_orig_idx = normal_idx[poor_local_idx]

        print(f"[HybridUS] OCSVM on normal class -> core: {n_core}, poorly-distributed: {n_poor}")

        # -- Phase 2: RandomUnderSampler on normal core only ------------------
        # total = n_fraud + n_core_undersampled + n_poor; solve for target ratio:
        #   n_core_target = n_fraud*(1-target)/target - n_poor
        n_core_target = (
            int(n_fraud * (1.0 - self.target_fraud_ratio) / self.target_fraud_ratio)
            - n_poor
        )

        if n_core_target <= 0:
            # Edge case (documented): protected poorly-distributed normals alone
            # already exceed the budget; the 5% target is unreachable without
            # removing protected samples. Keep all core normals.
            print(
                f"[HybridUS] WARNING: poorly-distributed normals ({n_poor}) "
                f"exceed the undersampling budget (n_core_target={n_core_target}). "
                f"Target fraud ratio unreachable without removing protected samples. "
                f"Keeping all core normals."
            )
            kept_core_local_idx = core_local_idx

        elif n_core_target >= n_core:
            print(
                f"[HybridUS] Normal core ({n_core}) already at or below target "
                f"({n_core_target}); skipping RandomUnderSampler."
            )
            kept_core_local_idx = core_local_idx

        else:
            X_rus = np.vstack([X_normal[core_mask], X_fraud])
            y_rus = np.concatenate(
                [np.zeros(n_core, dtype=int), np.ones(n_fraud, dtype=int)]
            )

            rus = RandomUnderSampler(
                sampling_strategy={0: n_core_target},
                random_state=self.random_state,
            )
            rus.fit_resample(X_rus, y_rus)

            kept_in_rus = rus.sample_indices_[rus.sample_indices_ < n_core]
            kept_core_local_idx = core_local_idx[kept_in_rus]

            print(
                f"[HybridUS] RandomUnderSampler: "
                f"core {n_core} -> {len(kept_core_local_idx)} samples kept"
            )

        # -- Combine - preserving temporal order ------------------------------
        kept_normal_orig_idx = np.concatenate(
            [normal_idx[kept_core_local_idx], poor_orig_idx]
        )
        all_kept_idx = np.concatenate([fraud_idx, kept_normal_orig_idx])
        all_kept_idx = np.sort(all_kept_idx)  # restore temporal order

        X_res = X[all_kept_idx]
        y_res = y[all_kept_idx]

        n_fraud_final = int(np.sum(y_res == config.FRAUD_LABEL))
        n_normal_final = int(np.sum(y_res == config.NORMAL_LABEL))
        actual_ratio = n_fraud_final / len(y_res)

        print(
            f"[HybridUS] Output - normal: {n_normal_final}, fraud: {n_fraud_final}  "
            f"(fraud ratio: {actual_ratio:.4f}, target: {self.target_fraud_ratio:.4f})"
        )

        return X_res, y_res
