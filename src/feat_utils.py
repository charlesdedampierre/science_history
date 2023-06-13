import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def gini(x):
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x) ** 2 * np.mean(x))


def normalize_cosine(data: pd.DataFrame, global_filter: float = 0.2):
    pivot = data.pivot("source", "target", "weight")
    pivot = pivot.fillna(0)
    similarity = cosine_similarity(pivot)
    df_sim = pd.DataFrame(similarity, index=pivot.index, columns=pivot.index)
    # df_sim = pd.DataFrame(similarity, index=pivot.index, columns=pivot.columns)
    df_sim["nodes"] = df_sim.index
    res = pd.melt(df_sim, id_vars=["nodes"]).sort_values("nodes")  # time
    res = res.dropna()
    res.columns = ["source", "target", "weight"]
    res = res[res["weight"] >= global_filter]
    res = res.sort_values(["source", "weight"], ascending=(False, False)).reset_index(
        drop=True
    )
    res["rank_cosine"] = res.groupby(["source"])["weight"].rank(
        method="first", ascending=False
    )

    return res


def specificity(
    df: pd.DataFrame, X: str, Y: str, Z: str, top_n: int = 50
) -> pd.DataFrame:
    if Z is None:
        Z = "count_values"
        df[Z] = 1
        group = df.groupby([X, Y]).agg(count_values=(Z, "sum")).reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()

    else:
        group = df.groupby([X, Y])[Z].sum().reset_index()
        cont = group.pivot(index=X, columns=Y, values=Z).fillna(0).copy()

    tx = df[X].value_counts()
    ty = df[Y].value_counts()

    cont = cont.astype(int)

    tx_df = pd.DataFrame(tx)
    tx_df.columns = ["c"]
    ty_df = pd.DataFrame(ty)
    ty_df.columns = ["c"]

    # Valeurs totales observées
    n = group[Z].sum()

    # Produit matriciel. On utilise pd.T pour pivoter une des deux séries.
    indep = tx_df.dot(ty_df.T) / n

    cont = cont.reindex(indep.columns, axis=1)
    cont = cont.reindex(indep.index, axis=0)

    # Contingency Matrix
    ecart = (cont - indep) ** 2 / indep
    chi2 = ecart.sum(axis=1)
    chi2 = chi2.sort_values(ascending=False)
    spec = ecart * np.sign(cont - indep)

    # Edge Table of X, Y, specificity measure
    spec[X] = spec.index
    edge = pd.melt(spec, id_vars=[X])
    edge.columns = [X, Y, "specificity"]
    edge = edge.sort_values(by=[X, "specificity"], ascending=[True, False]).reset_index(
        drop=True
    )
    edge = edge[edge["specificity"] > 0]
    edge = edge.groupby([X]).head(top_n)
    edge = edge.reset_index(drop=True)

    return edge
