import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans

main_path = '/home/paul/Desktop/cosi_importanti/uni/DS/sem_1/ML1/ml1_assignment3'
df = pd.read_csv(os.path.join(main_path, 'D.csv.xls'))

# Define features and their expected number of clusters
features_3_clusters = ['feature_0', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_7', 'feature_11']
features_2_clusters = ['feature_1', 'feature_6', 'feature_9']
all_features = features_3_clusters + features_2_clusters

# Filter to only features that exist in your dataframe
available_features = [f for f in all_features if f in df.columns]

print("=" * 80)
print("FEATURE VALUE CLUSTER DISTRIBUTION")
print("=" * 80)

results = {}

for feature in available_features:
    # Determine number of clusters
    if feature in features_3_clusters:
        n_clusters = 3
    else:
        n_clusters = 2

    # Use KMeans to find the cluster centers
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df[[feature]])

    # Get cluster centers and sort them
    centers = kmeans.cluster_centers_.flatten()
    sorted_idx = np.argsort(centers)

    # Remap labels to sorted order
    label_map = {old: new for new, old in enumerate(sorted_idx)}
    cluster_labels = np.array([label_map[l] for l in cluster_labels])
    centers_sorted = centers[sorted_idx]

    # Count samples per cluster
    counts = []
    percentages = []
    for i in range(n_clusters):
        count = np.sum(cluster_labels == i)
        percent = (count / len(df)) * 100
        counts.append(count)
        percentages.append(percent)

    results[feature] = {
        'n_clusters': n_clusters,
        'centers': centers_sorted,
        'counts': counts,
        'percentages': percentages
    }

    print(f"\n{feature}:")
    print(f"  Number of distinct value clusters: {n_clusters}")
    for i in range(n_clusters):
        print(f"    Cluster {i + 1}: center = {centers_sorted[i]:.4f}, count = {counts[i]} ({percentages[i]:.1f}%)")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)

summary_data = []
for feature in available_features:
    for i in range(results[feature]['n_clusters']):
        summary_data.append({
            'Feature': feature,
            'Cluster': i + 1,
            'Center_Value': results[feature]['centers'][i],
            'Count': results[feature]['counts'][i],
            'Percentage': f"{results[feature]['percentages'][i]:.1f}%"
        })

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# For features with 2 clusters, show which class is in which cluster
print("\n" + "=" * 80)
print("CLUSTER BY CLASS (for 2-cluster features)")
print("=" * 80)

for feature in available_features:
    if feature in features_2_clusters:
        # Get cluster labels
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(df[[feature]])

        # Sort clusters by mean value
        centers = kmeans.cluster_centers_.flatten()
        low_cluster = 0 if centers[0] < centers[1] else 1
        high_cluster = 1 - low_cluster

        print(f"\n{feature}:")
        for class_label in sorted(df['label'].unique()):
            class_mask = df['label'] == class_label
            class_clusters = cluster_labels[class_mask]

            low_count = np.sum(class_clusters == low_cluster)
            high_count = np.sum(class_clusters == high_cluster)
            total = low_count + high_count

            print(
                f"  Class {class_label}: Low value cluster: {low_count} ({low_count / total * 100:.1f}%), High value cluster: {high_count} ({high_count / total * 100:.1f}%)")