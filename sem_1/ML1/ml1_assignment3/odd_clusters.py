import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb

main_path = '/home/paul/Desktop/cosi_importanti/uni/DS/sem_1/ML1/ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv.xls'))
X = df.copy()
ids = X.pop('id')
labels = X.pop('label')
y = labels.copy()
cv = StratifiedKFold(n_splits=5, shuffle=True)


def micro_mcc(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    r = cm.shape[0]
    n = np.sum(cm)
    sum_TP = np.trace(cm)
    return (r * sum_TP - n) / (n * (r - 1))


scorer = make_scorer(micro_mcc)


def get_rare_cluster_mask(X, feature, n_clusters=2):
    """Get mask for items in the smallest (rarest) cluster of a feature"""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    cluster_labels = kmeans.fit_predict(X[[feature]])

    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)
    label_map = {old: new for new, old in enumerate(sorted_idx)}
    cluster_labels_sorted = np.array([label_map[l] for l in cluster_labels])

    cluster_sizes = [np.sum(cluster_labels_sorted == i) for i in range(n_clusters)]
    smallest_cluster = np.argmin(cluster_sizes)

    return cluster_labels_sorted == smallest_cluster


# Get masks for features 1, 6, 9
mask_f1 = get_rare_cluster_mask(X, 'feature_1', n_clusters=2)
mask_f6 = get_rare_cluster_mask(X, 'feature_6', n_clusters=2)
mask_f9 = get_rare_cluster_mask(X, 'feature_9', n_clusters=2)

# Define preprocessing cases
preprocessing_cases = {

    '6) Drop feature_1 & feature_9 rare': ~(mask_f1 | mask_f9),
}


# Merge all cases
all_cases = {**preprocessing_cases}

# Define models
scaler = StandardScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
for it in range(10):
    for hl in [(8, 8, 8, 8), (16,8), (64, 32, 16), (256,128, 64), (512,256, 16), (64, 16, 64), (64,32,16), (128,64,32), (512, 256, 128), (32, 32, 32, 32)]:

        models = {
            'Baseline': KNeighborsClassifier(n_neighbors=50),
            'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=250),
            'XGBoost': [xgb.XGBClassifier(objective='multi:softmax', num_class=2), X],
            'MLP': MLPClassifier(hidden_layer_sizes=hl, max_iter=800),
        }

        # Run all cases
        print("=" * 100)
        print(f"{hl}")
        print("=" * 100)

        all_results = {}

        for case_name, mask in all_cases.items():
            print(f"\n{'=' * 80}")
            print(f"CASE: {case_name}")
            print(f"{'=' * 80}")

            # Apply mask if specified
            if mask is not None:
                X_processed = X[mask]
                y_processed = y[mask]
                X_norm_processed = X_normalized[mask]
                print(f"Samples after filtering: {len(X_processed)} ({len(X_processed) / len(X) * 100:.1f}%)")
            else:
                X_processed = X
                y_processed = y
                X_norm_processed = X_normalized
                print(f"Samples: {len(X_processed)}")

            case_results = {}

            for model_name, model in models.items():
                # Choose which data to use (normalized for MLP, original for others)
                if model_name == 'MLP':
                    data = X_norm_processed
                else:
                    data = X_processed

                # Skip if too few samples
                if len(data) < 10:
                    print(f"  {model_name}: SKIPPED (too few samples)")
                    case_results[model_name] = None
                    continue

                try:
                    scores = cross_val_score(model, data, y_processed, cv=cv, scoring=scorer, n_jobs=-1)
                    mean_mcc = scores.mean()
                    std_mcc = scores.std()
                    print(f"  {model_name:15} | MCC = {mean_mcc:.4f} (+/- {std_mcc:.4f})")
                    case_results[model_name] = {'mean': mean_mcc, 'std': std_mcc, 'scores': scores}
                except Exception as e:
                    print(f"  {model_name}: ERROR - {str(e)[:50]}")
                    case_results[model_name] = None

            all_results[case_name] = case_results

        # Summary table
        print("\n" + "=" * 100)
        print("SUMMARY TABLE - BEST MCC PER CASE")
        print("=" * 100)

        summary_data = []
        for case_name, case_results in all_results.items():
            best_mcc = -1
            best_model = None
            for model_name, results in case_results.items():
                if results is not None and results['mean'] > best_mcc:
                    best_mcc = results['mean']
                    best_model = model_name
            if best_mcc > -1:
                summary_data.append({
                    'Case': case_name[:50],
                    'Top Model': best_model,
                    'Top MCC': f"{best_mcc:.4f}"
                })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

        # Find best overall
        print("\n" + "=" * 100)
        print("TOP OVERALL PERFORMANCE")
        print("=" * 100)

        best_overall = max([(case_results[model]['mean'], case_name, model)
                            for case_name, case_results in all_results.items()
                            for model, results in case_results.items()
                            if results is not None], default=None)

        if best_overall:
            mcc, case, model = best_overall
            print(f"Best: {case} | {model} | MCC = {mcc:.4f}")