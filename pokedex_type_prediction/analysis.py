import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, normalize

BASE = Path(__file__).resolve().parent
DATA, IMAGES, DOCS, RESULTS = (BASE / d for d in ("data", "images", "docs", "results"))
for d in (DATA, IMAGES, DOCS, RESULTS):
    d.mkdir(exist_ok=True)

type_colors = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0",
    "electric": "#F7D02C", "grass": "#7AC74C", "ice": "#96D9D6",
    "fighting": "#C22E28", "poison": "#A33EA1", "ground": "#E2BF65",
    "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC",
    "dark": "#705746", "steel": "#B7B7CE", "fairy": "#D685AD",
}

def rgba(hex_, a):
    h = hex_.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"

# data and embeddings
df = pd.read_csv(DATA / "pokemon_all.csv", keep_default_na=False)
n = len(df)

emb_path = DATA / "embeddings_all.npy"
if emb_path.exists() and len(np.load(emb_path)) == n:
    embeddings = np.load(emb_path)
else:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    np.save(emb_path, embeddings)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["types"].str.split("|"))
classes = np.array(mlb.classes_)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(clf, embeddings, Y, cv=kf)
proba = cross_val_predict(clf, embeddings, Y, cv=kf, method="predict_proba")

rep = classification_report(Y, pred, target_names=classes, zero_division=0, output_dict=True)
metrics = pd.DataFrame(rep).T.loc[list(classes)].reset_index().rename(columns={"index": "type"})
metrics.to_csv(RESULTS / "metrics_by_type.csv", index=False)

top3 = np.argsort(-proba, axis=1)[:, :3]
rows = np.arange(n)
pred_df = pd.DataFrame({
    "id": df["id"], "name": df["name"], "gen": df["gen"],
    "true_types": df["types"],
    "top1": classes[top3[:, 0]], "prob_top1": proba[rows, top3[:, 0]].round(3),
    "top2": classes[top3[:, 1]], "top3": classes[top3[:, 2]],
    "top1_correct": Y[rows, top3[:, 0]] == 1,
    "text": df["text"],
})
pred_df.to_csv(RESULTS / "predictions.csv", index=False)

hit = pred_df["top1_correct"].mean()
base = Y[:, list(classes).index("water")].mean()
by_gen = pred_df.groupby("gen")["top1_correct"].mean().round(3)
by_gen.to_csv(RESULTS / "accuracy_by_generation.csv")

E = normalize(embeddings)
S = E @ E.T
np.fill_diagonal(S, -np.inf)
K = 10
nn = np.argsort(-S, axis=1)[:, :K]

coh = []
for j, t in enumerate(classes):
    members = Y[:, j] == 1
    obs = Y[nn[members]][:, :, j].mean()
    chance = Y[:, j].mean()
    coh.append({"type": t, "neighbours_same_type": obs, "chance": chance, "lift": obs / chance})
coh = pd.DataFrame(coh).sort_values("lift", ascending=False)
coh.to_csv(RESULTS / "cohesion_by_type.csv", index=False)
lift = dict(zip(coh["type"], coh["lift"]))

Ec = E - E.mean(axis=0)
cent = normalize(np.vstack([Ec[Y[:, j] == 1].mean(axis=0) for j in range(len(classes))]))
C = pd.DataFrame(cent @ cent.T, index=classes, columns=classes)
C.round(3).to_csv(RESULTS / "type_similarity.csv")

plt.style.use("dark_background")

def barh(labels, values, title, xlabel, name, vline=None):
    labels, values = np.array(labels), np.array(values)
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels[order], values[order], color=[type_colors[t] for t in labels[order]])
    if vline is not None:
        ax.axvline(vline, ls="--", color="white", lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(IMAGES / f"{name}.png", dpi=200)
    plt.close(fig)

barh(metrics["type"], metrics["f1-score"],
     "Which types does the Pokédex text give away? (F1 per type)", "F1 score", "f1_by_type")
barh(coh["type"], coh["lift"],
     f"Do text neighbours share a type? (K={K}, 1 = chance)", "Lift over chance",
     "cohesion_by_type", vline=1)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(C.values, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=90)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Type similarity according to Pokédex text")
fig.tight_layout()
fig.savefig(IMAGES / "type_similarity.png", dpi=200)
plt.close(fig)

def explorer(coords, dims, name):
    Scatter = go.Scatter if dims == 2 else go.Scatter3d
    axes = dict(x=coords[:, 0], y=coords[:, 1])
    if dims == 3:
        axes["z"] = coords[:, 2]
    custom = np.stack([
        df["name"], df["types"].str.replace("|", " / ", regex=False),
        df["text"].apply(lambda t: textwrap.fill(t, 60).replace("\n", "<br>")),
    ], axis=-1)
    all_cols = [rgba(type_colors[t], 0.9) for t in df["type1"]]

    fig = go.Figure(Scatter(
        mode="markers", **axes, customdata=custom,
        marker=dict(size=7 if dims == 3 else 8, color=all_cols),
        hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br>%{customdata[2]}<extra></extra>",
    ))

    buttons = [dict(label="All (colour = type 1)", method="restyle",
                    args=[{"marker.color": [all_cols], "marker.size": [[8] * n]}])]
    for j, t in enumerate(classes):
        mask = Y[:, j] == 1
        cols = [rgba(type_colors[t], 1) if m else "rgba(120,120,120,0.12)" for m in mask]
        sizes = [11 if m else 4 for m in mask]
        buttons.append(dict(label=f"{t} ({mask.sum()}) · lift x{lift[t]:.1f}", method="restyle",
                            args=[{"marker.color": [cols], "marker.size": [sizes]}]))

    fig.update_layout(
        template="plotly_dark", title="Explorer: filter by type (type 1 or type 2)",
        updatemenus=[dict(buttons=buttons, direction="down", x=0.01, y=1.1, xanchor="left")],
    )
    if dims == 2:
        fig.update_layout(xaxis_visible=False, yaxis_visible=False)
    else:
        fig.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    fig.write_html(DOCS / f"{name}.html")

explorer(umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42).fit_transform(embeddings),
         2, "explorer_2d")
explorer(umap.UMAP(n_neighbors=30, min_dist=0.3, n_components=3, random_state=42).fit_transform(embeddings),
         3, "explorer_3d")

with open(RESULTS / "summary.txt", "w") as f:
    f.write(f"Pokémon: {n}\n")
    f.write(f"Best guess is one of its real types: {hit:.1%} (always-water baseline: {base:.1%})\n")
    f.write(f"Macro F1: {metrics['f1-score'].mean():.2f}\n\n")
    f.write("Accuracy by generation:\n" + by_gen.to_string() + "\n\n")
    f.write(f"Cohesion (lift, K={K}):\n" + coh[["type", "lift"]].round(2).to_string(index=False) + "\n")

print(open(RESULTS / "summary.txt").read())