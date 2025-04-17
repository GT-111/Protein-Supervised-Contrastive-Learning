# compare_cluster_raw_vs_embedding.py (evaluate full set + test set)
import pandas as pd
import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from protein_embedding import SupConEncoder

df_full = pd.read_csv("protein_features.csv")
X_full = df_full.drop(columns=["pdb_id", "label", "subclass"]).values
y_full_label = df_full["label"].values
y_full_subclass = df_full["subclass"].values

X_test = pd.read_csv("X_test_scaled.csv").values
y_test_info = pd.read_csv("y_test_info.csv")
y_test_label = y_test_info["label"].values
y_test_subclass = y_test_info["subclass"].values

le = LabelEncoder()
y_full_sub_encoded = le.fit_transform(y_full_subclass)
y_test_sub_encoded = le.transform(y_test_subclass)

scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SupConEncoder(input_dim=X_full_scaled.shape[1], embed_dim=64).to(device)
model.load_state_dict(torch.load("protein_embedding.pt", map_location=device))
model.eval()
with torch.no_grad():
    X_full_tensor = torch.tensor(X_full_scaled, dtype=torch.float32).to(device)
    X_full_embed = model(X_full_tensor).cpu().numpy()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_test_embed = model(X_test_tensor).cpu().numpy()

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
from sklearn.manifold import TSNE

def cluster_then_project_and_color_by_subclass(X_raw, X_embed, y_sub_encoded, subclass_names, suffix="test"):
    n_clusters = len(np.unique(y_sub_encoded))

    def cluster_and_score(name, X):
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X)
        ari = adjusted_rand_score(y_sub_encoded, clusters)
        nmi = normalized_mutual_info_score(y_sub_encoded, clusters)
        hom = homogeneity_score(y_sub_encoded, clusters)
        print(f"[{name}] Clustering: ARI={ari:.4f}, NMI={nmi:.4f}, Homogeneity={hom:.4f}")
        return clusters

    clusters_raw = cluster_and_score("Raw", X_raw)
    clusters_embed = cluster_and_score("SupCon", X_embed)

    proj_raw = TSNE(n_components=2, random_state=42).fit_transform(X_raw)
    proj_embed = TSNE(n_components=2, random_state=42).fit_transform(X_embed)

    def build_df(Z, clusters):
        df = pd.DataFrame(Z, columns=["x", "y"])
        df["subclass"] = [subclass_names[i] for i in y_sub_encoded]
        df["cluster"] = clusters
        return df

    df_raw = build_df(proj_raw, clusters_raw)
    df_embed = build_df(proj_embed, clusters_embed)

    datasets = [
        ("Raw Features", df_raw),
        ("SupCon Embedding", df_embed)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for col, (title, df) in enumerate(datasets):
        # Top row: colored by subclass
        sns.scatterplot(data=df, x="x", y="y", hue="subclass", palette="tab20", ax=axes[0, col], s=30, legend=False)
        axes[0, col].set_title(f"{title} (Colored by Subclass)", fontsize=12)
        axes[0, col].axis("off")

        # Bottom row: colored by cluster
        sns.scatterplot(data=df, x="x", y="y", hue="cluster", palette="tab10", ax=axes[1, col], s=30, legend=False)
        axes[1, col].set_title(f"{title} (Colored by Cluster)", fontsize=12)
        axes[1, col].axis("off")

    plt.suptitle("Clustering → t-SNE → Visualization (Test Set)", fontsize=16)
    plt.tight_layout()
    plt.savefig("panel_test_cluster_by_subclass_and_cluster.png", dpi=300)
    plt.close()


cluster_then_project_and_color_by_subclass(
    X_raw=X_test,
    X_embed=X_test_embed,
    y_sub_encoded=y_test_sub_encoded,
    subclass_names=le.classes_,
    suffix="test"
)
