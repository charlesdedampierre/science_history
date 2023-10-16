import pandas as pd
import polars as pl

from .feat_utils import normalize_cosine, specificity


def get_edge_node_table(df: pl.DataFrame, sample=10000, seed=42):
    # Create a node table
    df_nodes = df.to_pandas().groupby(["target"])["weight"].sum().reset_index()
    df_nodes.columns = ["node", "sum_weight"]
    df_nodes = df_nodes.sort_values("sum_weight", ascending=False)
    df_nodes = df_nodes.reset_index(drop=True)

    df = df.to_pandas()
    df["weight"] = 1
    df = pl.from_pandas(df)

    matrix = df.pivot(index="source", columns="target", values="weight")
    matrix = matrix.fill_null(0)
    matrix = matrix.to_pandas()
    matrix = matrix.set_index("source")

    if sample <= len(matrix):
        matrix = matrix.sample(
            n=sample, random_state=seed
        )  # Specify the number of rows to sample

    # Compute the co-occurrence matrix by multiplying the matrix by its transpose
    cooc = matrix.T.dot(matrix)

    co_occurrence = cooc.unstack().reset_index()
    co_occurrence.columns = ["source", "target", "weight"]

    df_cooc = co_occurrence[co_occurrence["weight"] > 0].reset_index(drop=True)
    df_cooc = df_cooc[df_cooc["source"] != df_cooc["target"]].reset_index(drop=True)

    # Compute Count Ranking
    df_cooc = df_cooc.sort_values(
        ["source", "weight"], ascending=(False, False)
    ).reset_index(drop=True)
    df_cooc["rank_count"] = df_cooc.groupby(["source"])["weight"].rank(
        method="first", ascending=False
    )

    # Compute Specificity Ranking
    df_spec = specificity(df_cooc, X="source", Y="target", Z="weight", top_n=1000)
    df_edge = pd.merge(df_spec, df_cooc, on=["source", "target"])

    df_edge = df_edge.sort_values(
        ["source", "specificity"], ascending=(False, False)
    ).reset_index(drop=True)
    df_edge["rank_specificity"] = df_edge.groupby(["source"])["specificity"].rank(
        method="first", ascending=False
    )

    return df_edge, df_nodes


def filter_edge_table(
    df_edge: pd.DataFrame,
    edge_rule: str = "specificity",
    top_directed_neighbours: int = 5,
    normalize_on_top: bool = True,
    min_count_link=10,
) -> pd.DataFrame:
    df_edge = df_edge[df_edge["weight"] >= min_count_link]
    df_edge = df_edge[df_edge["source"] != df_edge["target"]]
    if edge_rule == "specificity":
        # the weight is now the chi2
        df_filter = df_edge.drop("weight", axis=1)
        df_filter = df_filter.rename(columns={"specificity": "weight"})
        df_filter = df_filter[df_filter["rank_specificity"] <= top_directed_neighbours]

        if normalize_on_top:
            df_filter = normalize_cosine(df_filter)
            df_filter = df_filter[df_filter["rank_cosine"] <= top_directed_neighbours]

    elif edge_rule == "count":
        df_filter = df_edge[df_edge["rank_count"] <= top_directed_neighbours]

        if normalize_on_top:
            df_filter = normalize_cosine(df_filter)
            df_filter = df_filter[df_filter["rank_cosine"] <= top_directed_neighbours]

    elif edge_rule == "cosine":
        df_filter = normalize_cosine(df_edge)
        df_filter = df_filter[df_filter["rank_cosine"] <= top_directed_neighbours]

    return df_filter
