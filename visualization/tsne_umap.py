import umap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embeddings(embeddings, labels, output_path, method='tsne'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
    projections = reducer.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=labels, palette='tab10', s=15, alpha=0.8)
    plt.title(f'{method.upper()} Projection')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
