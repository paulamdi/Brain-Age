import itertools
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer

# ================= Config =================
IN_CSV = "FULL_FINE_TUNING_shap_embeddings.csv"  # <— sin espacio inicial
SEED = 42
OUT_DIR = "umap_grid_results"
os.makedirs(OUT_DIR, exist_ok=True)

SCALERS = {
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "quantile": QuantileTransformer(output_distribution="normal", random_state=SEED),
    "power": PowerTransformer(method="yeo-johnson", standardize=True),
    "none": None
}

# Para 32 dims de SHAP: None (=sin PCA), 5/10/20 (=reduce), 32 (=como identidad)
PCA_DIMS = [None, 5, 10, 20, 32]
PCA_WHITEN = [False, True]  # si pca_dim es None, saltamos whiten=True

# UMAP params grid
N_NEIGHBORS = [10, 15, 30]
MIN_DIST = [0.0, 0.1, 0.3]
METRICS = ["euclidean", "cosine", "correlation"]
DENSMAP = [False, True]

K_RANGE = range(2, 9)  # silhouette sweep

# ============== Load embeddings ==============
df = pd.read_csv(IN_CSV)
embed_cols = [c for c in df.columns if c.startswith("embed_")]
assert len(embed_cols) > 0, "No 'embed_*' columns found."
X_raw = df[embed_cols].values
subj = df["Subject_ID"].values

n_samples, n_features = X_raw.shape
max_pca = min(n_samples, n_features)

def preprocess(X, scaler_name, pca_dim, pca_whiten):
    # 1) scale
    if scaler_name != "none":
        scaler = SCALERS[scaler_name]
        Xs = scaler.fit_transform(X)
    else:
        Xs = X

    # 2) PCA (optional)
    if pca_dim is not None:
        if pca_dim > max_pca:
            raise ValueError(f"PCA n_components={pca_dim} > max permitted ({max_pca}).")
        pca = PCA(n_components=pca_dim, whiten=pca_whiten, random_state=SEED)
        Xp = pca.fit_transform(Xs)
    else:
        Xp = Xs

    return Xp

def choose_best_k(X_proc):
    best_k, best_sil = None, -np.inf
    best_labels = None
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = km.fit_predict(X_proc)
        sil = silhouette_score(X_proc, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    return best_k, best_sil, best_labels

results = []
best_global = {"silhouette": -np.inf, "tag": None}
proc_cache = {}  # cache para (scaler, pca_dim, whiten) -> X_proc

# ============== Grid search ==============
combo_id = 0
for scaler_name, pca_dim, pca_whiten in itertools.product(SCALERS.keys(), PCA_DIMS, PCA_WHITEN):
    # Skip redundante: si no hay PCA, whiten no aplica
    if pca_dim is None and pca_whiten is True:
        continue

    # Limitar PCA=32 (idéntico a no reducir) y evitar > max_pca
    if pca_dim is not None and pca_dim > max_pca:
        continue

    key = (scaler_name, pca_dim, pca_whiten)
    if key in proc_cache:
        X_proc = proc_cache[key]
    else:
        X_proc = preprocess(X_raw, scaler_name, pca_dim, pca_whiten)
        proc_cache[key] = X_proc

    best_k, best_sil, labels = choose_best_k(X_proc)

    for n_nb, md, metric, dens in itertools.product(N_NEIGHBORS, MIN_DIST, METRICS, DENSMAP):
        combo_id += 1
        tag = f"scaler={scaler_name}|pca={pca_dim}|whiten={pca_whiten}|nn={n_nb}|minD={md}|metric={metric}|dens={dens}|k={best_k}"
        print(f"[{combo_id:03d}] {tag}  | silhouette={best_sil:.3f}")

        reducer = umap.UMAP(
            n_neighbors=n_nb,
            min_dist=md,
            metric=metric,
            densmap=dens,
            random_state=SEED  # fija resultados; si quieres paralelismo, quita el seed
        )
        X_umap = reducer.fit_transform(X_proc)

        plot_df = pd.DataFrame({
            "UMAP1": X_umap[:,0],
            "UMAP2": X_umap[:,1],
            "Cluster": labels,
            "Subject_ID": subj
        })

        plt.figure(figsize=(7,6))
        sns.scatterplot(
            data=plot_df, x="UMAP1", y="UMAP2", hue="Cluster",
            palette="Set2", s=50, edgecolor="k", linewidth=0.2
        )
        plt.title(f"UMAP + KMeans | {tag}\nSilhouette={best_sil:.3f}")
        plt.tight_layout()
        fname_base = tag.replace("|", "__").replace(" ", "")
        plt.savefig(os.path.join(OUT_DIR, f"{fname_base}.png"), dpi=250)
        plt.close()

        out_csv = os.path.join(OUT_DIR, f"{fname_base}.csv")
        plot_df.to_csv(out_csv, index=False)

        if best_sil > best_global["silhouette"]:
            best_global = {
                "silhouette": best_sil,
                "tag": tag,
                "csv": out_csv,
                "png": os.path.join(OUT_DIR, f"{fname_base}.png")
            }

        results.append({
            "scaler": scaler_name,
            "pca_dim": pca_dim,
            "pca_whiten": pca_whiten,
            "n_neighbors": n_nb,
            "min_dist": md,
            "metric": metric,
            "densmap": dens,
            "best_k": best_k,
            "silhouette": best_sil,
            "csv": out_csv
        })

df_res = pd.DataFrame(results).sort_values(by="silhouette", ascending=False)
df_res.to_csv(os.path.join(OUT_DIR, "umap_grid_summary.csv"), index=False)

print("\n=== BEST by silhouette (on preprocessed space) ===")
print(f"silhouette={best_global['silhouette']:.3f}")
print(f"combo    = {best_global['tag']}")
print(f"CSV      = {best_global['csv']}")
print(f"FIG      = {best_global['png']}")
print(f"\nSummary saved to: {os.path.join(OUT_DIR, 'umap_grid_summary.csv')}")
