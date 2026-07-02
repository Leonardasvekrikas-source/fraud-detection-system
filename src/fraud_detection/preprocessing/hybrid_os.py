# =============================================================================
# preprocessing/hybrid_os.py
#
# HybridOS  -  paper Section 2.3, Subsystem 1
#
# Distribution-preserving oversampling method for the FRAUD class.
# Used in the non-sequential (LightGBM) subsystem.
#
# Algorithm (Fig. 2 - left column):
#   Phase 1  OCSVM on fraud class -> "fraud core" (+1) vs "fraud outliers" (-1).
#            Outliers are set aside (not removed).
#   Phase 2  SMOTE on fraud core only -> synthetic fraud within the core.
#   Combine  normal + synthetic_fraud_core + fraud_core + fraud_outliers.
#
# Target: fraud = TARGET_FRAUD_RATIO (5%) of the combined training set.
# =============================================================================

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.svm import OneClassSVM

from fraud_detection import config


class HybridOS:
    """HybridOS: distribution-preserving oversampling (paper Section 2.3)."""

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
        """Apply HybridOS to the training fold. Returns (X_res, y_res)."""
        X = np.asarray(X_train)
        y = np.asarray(y_train)

        fraud_mask = y == config.FRAUD_LABEL
        normal_mask = y == config.NORMAL_LABEL

        X_fraud = X[fraud_mask]
        X_normal = X[normal_mask]

        n_fraud = len(X_fraud)
        n_normal = len(X_normal)

        print(f"[HybridOS] Input - normal: {n_normal}, fraud: {n_fraud}")

        # -- Phase 1: OCSVM on fraud class ------------------------------------
        ocsvm = OneClassSVM(kernel=self.ocsvm_kernel, nu=self.ocsvm_nu)
        ocsvm.fit(X_fraud)
        fraud_ocsvm_labels = ocsvm.predict(X_fraud)  # +1 or -1

        core_mask = fraud_ocsvm_labels == 1
        outlier_mask = fraud_ocsvm_labels == -1

        X_fraud_core = X_fraud[core_mask]
        X_fraud_outliers = X_fraud[outlier_mask]

        n_core = len(X_fraud_core)
        n_outliers = len(X_fraud_outliers)

        print(f"[HybridOS] OCSVM on fraud class -> core: {n_core}, outliers: {n_outliers}")

        # -- Phase 2: SMOTE on fraud core only --------------------------------
        n_fraud_new_target = int(
            self.target_fraud_ratio * n_normal / (1.0 - self.target_fraud_ratio)
        )

        if n_core >= n_fraud_new_target:
            print(
                f"[HybridOS] Fraud core ({n_core}) already meets target "
                f"({n_fraud_new_target}); skipping SMOTE."
            )
            X_fraud_resampled = X_fraud_core
        else:
            smote_ratio = n_fraud_new_target / n_normal
            k_neighbors = min(5, n_core - 1)
            if k_neighbors < 1:
                print(
                    f"[HybridOS] Too few fraud core samples ({n_core}) for SMOTE; "
                    f"using fraud core as-is."
                )
                X_fraud_resampled = X_fraud_core
            else:
                X_smote = np.vstack([X_normal, X_fraud_core])
                y_smote = np.concatenate(
                    [
                        np.full(n_normal, config.NORMAL_LABEL),
                        np.full(n_core, config.FRAUD_LABEL),
                    ]
                )

                smote = SMOTE(
                    sampling_strategy=smote_ratio,
                    k_neighbors=k_neighbors,
                    random_state=self.random_state,
                )
                X_smote_res, y_smote_res = smote.fit_resample(X_smote, y_smote)

                fraud_res_mask = y_smote_res == config.FRAUD_LABEL
                X_fraud_resampled = X_smote_res[fraud_res_mask]

                print(
                    f"[HybridOS] SMOTE expanded fraud core "
                    f"{n_core} -> {len(X_fraud_resampled)} samples "
                    f"(target ratio {smote_ratio:.4f})"
                )

        # -- Combine: normal + resampled fraud + fraud outliers ---------------
        X_res = np.vstack([X_normal, X_fraud_resampled, X_fraud_outliers])
        y_res = np.concatenate(
            [
                np.full(len(X_normal), config.NORMAL_LABEL),
                np.full(len(X_fraud_resampled), config.FRAUD_LABEL),
                np.full(len(X_fraud_outliers), config.FRAUD_LABEL),
            ]
        )

        n_fraud_final = int(np.sum(y_res == config.FRAUD_LABEL))
        n_normal_final = int(np.sum(y_res == config.NORMAL_LABEL))
        actual_ratio = n_fraud_final / len(y_res)

        print(
            f"[HybridOS] Output - normal: {n_normal_final}, fraud: {n_fraud_final}  "
            f"(fraud ratio: {actual_ratio:.4f}, target: {self.target_fraud_ratio:.4f})"
        )

        return X_res, y_res
