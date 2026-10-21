import pandas as pd
import numpy as np
import os
import joblib
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

main_path = '/ml1_assignment3'

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv(os.path.join(main_path, 'D.csv.xls'))
X = df.copy()
ids = X.pop('id')
labels = X.pop('label')
y = labels.copy()

df_out = pd.read_csv(os.path.join(main_path, 'D_out.csv.xls'))
X_out = df_out.copy()
if 'id' in X_out.columns:
    X_out = X_out.drop(columns=['id'])
if 'label' in X_out.columns:
    y_out = X_out.pop('label')
# ============================================================================
# STEP 1: TRAIN GMM ON OUTLIERS (3 components)
# ============================================================================
scaler = StandardScaler()
X_out_scaled = scaler.fit_transform(X_out)

gmm = GaussianMixture(
    n_components=3,

    covariance_type='full',
    max_iter=1000,
    tol=1e-3
)
gmm.fit(X_out_scaled)
train_scores = np.exp(gmm.score_samples(X_out_scaled))
threshold = np.percentile(train_scores, 50)
# ============================================================================
# STEP 2: IDENTIFY OUTLIERS IN MAIN DATA
# ============================================================================

X_scaled = scaler.transform(X)
all_scores = np.exp(gmm.score_samples(X_scaled))

outlier_mask = all_scores > threshold
n_outliers = outlier_mask.sum()
n_kept = (~outlier_mask).sum()
# ============================================================================
# STEP 3: CREATE FILTERED DATASET (X_in = X without outliers)
# ============================================================================
X_in = X[~outlier_mask]
y_in = y[~outlier_mask]
# ============================================================================
# MACRO MCC FUNCTION
# ============================================================================
def macro_mcc(y_true, y_pred):
    """
    Macro-averaged Matthews Correlation Coefficient
    Computes MCC per class (one-vs-all) and averages
    """
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]

    mcc_per_class = []

    for i in range(r):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)

        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator

        mcc_per_class.append(mcc)

    return np.mean(mcc_per_class)
# ============================================================================
# STEP 4: TRAIN BALANCED RANDOM FOREST ON X_in
# ============================================================================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    class_weight='balanced',
    n_jobs=-1
)
rf.fit(X_in, y_in)


def predict_with_outlier_check(X_input, print_outliers=True):
    """
    Convenience function: Takes features X, returns predictions with outlier checking.

    Parameters:
    - X_input: DataFrame or array of features (n_samples, n_features)
    - print_outliers: bool, whether to print outlier status

    Returns:
    - y_pred: array of predicted class labels
    """
    X_scaled = scaler.transform(X_input)
    scores = np.exp(gmm.score_samples(X_scaled))
    outlier_flags = scores > threshold
    y_pred = rf.predict(X_input)

    if print_outliers:
        for i in range(len(y_pred)):
            status = "⚠️ OUTLIER" if outlier_flags[i] else "✅ Normal"
            print(f"Sample {i}: Predicted Class = {y_pred[i]}, Status = {status}, Score = {scores[i]:.6f}")
