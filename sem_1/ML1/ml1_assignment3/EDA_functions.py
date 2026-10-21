import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from itertools import permutations


plt.style.use('dark_background')
df_full = pd.read_csv('/ml1_assignment3/D.csv.xls')
X = df_full.copy()
y = df_full['label']
labels = X.pop('label')

def plot_feature_by_class_boxplots(df_full, feature_names, class_col='label'):
    """
    Create separate boxplot for each feature showing distribution across classes

    Parameters:
    - df_full: DataFrame with class and features
    - feature_names: list of 12 feature columns to plot
    - class_col: name of the class column
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.ravel()

    for idx, feature in enumerate(feature_names):
        # Create boxplot
        sns.boxplot(data=df_full, x=class_col, y=feature, ax=axes[idx],
                    palette='tab10')
        # Add swarmplot for individual points (optional, can be slow)
        # sns.swarmplot(data=df_full, x=class_col, y=feature, ax=axes[idx],
        #              color='black', alpha=0.3, size=2)

        axes[idx].set_title(f'Feature: {feature}', fontsize=12)
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions Across Classes', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_by_class_violin(df_full, feature_names, class_col='label'):
    """
    Violin plots - better for showing distribution shape
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.ravel()

    for idx, feature in enumerate(feature_names):
        sns.violinplot(data=df_full, x=class_col, y=feature, ax=axes[idx],
                       palette='tab10', inner='quartile')

        axes[idx].set_title(f'Feature: {feature}', fontsize=12)
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(feature)
        axes[idx].grid(True, alpha=0.3)

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Violin Plots: Feature Distributions Across Classes', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_by_class_line(df_full, feature_names, class_col='label'):
    """
    Line plot showing mean + confidence interval for each feature per class
    """
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.ravel()

    for idx, feature in enumerate(feature_names):
        # Calculate mean and std per class
        stats = df_full.groupby(class_col)[feature].agg(['mean', 'std', 'count']).reset_index()
        stats['sem'] = stats['std'] / np.sqrt(stats['count'])  # standard error

        # Line plot with error bars
        axes[idx].errorbar(x=stats[class_col], y=stats['mean'],
                           yerr=stats['sem'], fmt='o-', capsize=5,
                           markersize=8, linewidth=2, color='steelblue')

        axes[idx].set_title(f'Feature: {feature}', fontsize=12)
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(f'{feature} (mean ± SEM)')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks(range(len(stats[class_col])))
        axes[idx].set_xticklabels(stats[class_col])

    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Means Across Classes (with Standard Error)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_feature_heatmap_by_class(df_full, feature_names, class_col='label'):
    """
    Heatmap showing mean value of each feature for each class
    Best for overview - which features separate which classes
    """
    # Calculate mean of each feature per class
    class_means = df_full.groupby(class_col)[feature_names].mean()

    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(class_means.T, annot=True, fmt='.2f', cmap='coolwarm',
                cbar_kws={'label': 'Mean Value'}, linewidths=0.5)

    plt.title('Feature Means by Class (Normalized View)', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Also show normalized version (per feature scaling)
    plt.figure(figsize=(14, 8))
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    class_means_scaled = pd.DataFrame(
        scaler.fit_transform(class_means.T).T,
        index=class_means.index,
        columns=class_means.columns
    )
    sns.heatmap(class_means_scaled.T, annot=True, fmt='.2f', cmap='coolwarm',
                cbar_kws={'label': 'Z-score'}, linewidths=0.5)

    plt.title('Feature Means by Class (Z-scored per Feature)', fontsize=14)
    plt.xlabel('Class')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


# Usage examples:
feature_cols = X.columns.tolist()  # your 12 features

# Choose one or try all:
plot_feature_by_class_boxplots(df_full, feature_cols)  # Boxplots - good for outliers
plot_feature_by_class_violin(df_full, feature_cols)  # Violin - best for distribution shape
plot_feature_by_class_line(df_full, feature_cols)  # Line - best for seeing trends
plot_feature_heatmap_by_class(df_full, feature_cols)  # Heatmap - best overview






'''
def make_pca():
X    X = df_full.copy()
    y = df_full['label']
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Standardize (center + scale to unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Then apply PCA, 3 components gives 98.32%
    pca = PCA(n_components=3)
    pca_full = PCA()
    X_pca = pca.fit_transform(X_scaled)
    X_pca_full = pca_full.fit_transform(X_scaled)
    return X_pca, X_pca_full

def make_class_plots():
    def plot_class_distributions(df_full, class_label, feature_names):
        """
        Create 12 subplots showing distribution of each feature for a specific class

        Parameters:
        - df_full: DataFrame with 'label' column and features
        - class_label: the class to analyze (e.g., 'Class_A')
        - feature_names: list of 12 feature column names
        """
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()

        # Get data for the target class
        class_data = df_full[df_full['label'] == class_label]
        all_data = df_full  # for comparison

        for idx, feature in enumerate(feature_names):
            # Plot histogram with KDE
            axes[idx].hist(class_data[feature], bins=30, alpha=0.7,
                           color='red', label=f'{class_label}', density=True)
            axes[idx].hist(all_data[feature], bins=30, alpha=0.3,
                           color='white', label='All data', density=True)

            # Add KDE
            sns.kdeplot(data=class_data[feature], ax=axes[idx],
                        color='orange', linewidth=2)

            axes[idx].set_title(f'Feature: {feature}', fontsize=12)
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle(f'Feature Distributions for Class: {class_label}',
                     fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'class_{class_label}.png')
        plt.show()

    # Usage:
    feature_cols = [col for col in X.columns]  # your 12 features
    for label in range(5):
        plot_class_distributions(df_full, label, feature_cols)

def make_parallel_coor():
    def plot_parallel_coordinates(df_full, feature_names, class_col='label'):
        """
        Create parallel coordinates plot colored by class

        Parameters:
        - df_full: DataFrame with class and features
        - feature_names: list of feature columns to plot
        - class_col: name of the class column
        """
        # Normalize features for better visualization
        scaler = MinMaxScaler()
        df_normalized = df_full.copy()
        df_normalized[feature_names] = scaler.fit_transform(df_full[feature_names])

        # Create the plot
        plt.figure(figsize=(15, 8))
        parallel_coordinates(df_normalized, class_col,
                             cols=feature_names,
                             alpha=0.5,
                             linewidth=0.8,
                             colormap='tab10')  # handles up to 10 classes

        plt.title('Parallel Coordinates Plot (Features Normalized to [0,1])', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'parallel_coordinates_{class_col}.png')

    feature_cols = X.columns.tolist()  # your 12 features
    plot_parallel_coordinates(df_full, feature_cols)


def make_t_sne():
    def plot_tsne_2d(X, y, perplexity=30, random_state=42):
        """
        Create 2D t-SNE visualization colored by class

        Parameters:
        - X: feature matrix (9000, 12)
        - y: class labels (9000,)
        - perplexity: t-SNE parameter (typically 5-50)
        """
        # t-SNE can be slow on 9000 points, sample if needed
        if X.shape[0] > 5000:
            print(f"Large dataset ({X.shape[0]} points). Consider sampling for faster computation.")
            # Optional: random sample
            # sample_idx = np.random.choice(X.shape[0], 3000, replace=False)
            # X = X[sample_idx]
            # y = y[sample_idx]

        # Initialize and fit t-SNE
        print("Computing t-SNE... (may take a few minutes)")
        tsne = TSNE(n_components=2, random_state=random_state,
                    perplexity=perplexity, n_iter=1000)
        X_tsne = tsne.fit_transform(X)

        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=y.astype('category').cat.codes if hasattr(y, 'cat') else y,
                              cmap='tab10', alpha=0.6, s=10)

        plt.colorbar(scatter, label='Class')
        plt.title(f't-SNE Visualization (perplexity={perplexity})', fontsize=14)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.show()

        return X_tsne

    # Version with multiple perplexity values to try
    def plot_tsne_multiple(X, y, perplexities=[5, 30, 50]):
        """
        Try multiple perplexity values to find the best separation
        """
        fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))

        for idx, perp in enumerate(perplexities):
            print(f"Computing t-SNE with perplexity={perp}...")
            tsne = TSNE(n_components=2, random_state=42,
                        perplexity=perp, n_iter=1000)
            X_tsne = tsne.fit_transform(X)

            axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=y.astype('category').cat.codes if hasattr(y, 'cat') else y,
                              cmap='tab10', alpha=0.6, s=10)
            axes[idx].set_title(f'Perplexity = {perp}')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            axes[idx].grid(True, alpha=0.3)

        plt.suptitle('t-SNE with Different Perplexity Values', fontsize=14)
        plt.tight_layout()
        plt.show()

    # Usage:
    plot_tsne_2d(X, y, perplexity=30)
    plot_tsne_multiple(X, y, perplexities=[10, 30, 50])


'''