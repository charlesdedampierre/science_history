import sqlite3
import typing as t

import pandas as pd

pd.options.mode.chained_assignment = None


def clean_list_occupation(
    clean_dict: t.List[dict], list_targeted: t.List[str]
) -> t.List[str]:
    for list_model in clean_dict:
        occupation_pair = list_model["occupation_pair"]
        remove_occupation = list_model["remove_occupation"]

        if all(elem in list_targeted for elem in occupation_pair):
            list_targeted.remove(remove_occupation)
        else:
            pass

    return list_targeted


def check(list_targeted):
    if all(elem in list_targeted for elem in ["botanist", "biologist"]):
        return 1
    else:
        return 0


def check_if_contains(list, occupation: str):
    if occupation in list:
        return 1
    else:
        return 0


def get_proportionned_occupations(
    data_final, data_merge, occupation_select="biologist"
):
    list_inds = list(
        set(data_merge[data_merge["meta_occupation"] == occupation_select].wikidata_id)
    )
    data_select = data_merge[data_merge["wikidata_id"].isin(list_inds)]

    def get_len_occupation(x="zoologist"):
        len_occ_1 = len(
            set(data_select[data_select["meta_occupation"] == x].wikidata_id)
        )
        res = round(len_occ_1 / len(list_inds), 1)
        return res

    len_zoologist = get_len_occupation(x="zoologist")
    len_botanist = get_len_occupation(x="botanist")
    len_anatomist = get_len_occupation(x="anatomist")

    data_final_without_bio = data_final[
        data_final["meta_occupation"] != occupation_select
    ]
    data_final_biologists = data_final[
        data_final["meta_occupation"] == occupation_select
    ]

    number_zoologists = int(len_zoologist * len(data_final_biologists))
    number_botanists = int(len_botanist * len(data_final_biologists))
    number_anatomists = int(len_anatomist * len(data_final_biologists))

    data_final_biologists["meta_occupation"][:number_zoologists] = "zoologist"
    data_final_biologists["meta_occupation"][
        number_zoologists : number_zoologists + number_botanists
    ] = "botanist"
    data_final_biologists["meta_occupation"][
        number_zoologists + number_botanists :
    ] = "anatomist"

    df_final_new = pd.concat([data_final_without_bio, data_final_biologists])

    return df_final_new


if __name__ == "__main__":
    from functions.env import DATA_PATH, DB_SCIENCE_PATH

    data = pd.read_csv(DATA_PATH + "/df_indi_occupations.csv", index_col=[0])
    df_annotation = pd.read_excel(
        DATA_PATH + "/ENS - True Science.xlsx", sheet_name="cleaning_top_occupations"
    )
    df_annotation = df_annotation[df_annotation["erase"].isna()]
    df_annotation = df_annotation[df_annotation["count_occupation"] >= 10]
    df_annotation = df_annotation[["occupation", "meta_occupation"]].reset_index(
        drop=True
    )

    clean_dict = pd.read_excel(
        DATA_PATH + "/ENS - True Science.xlsx", sheet_name="co_occurence_occupation"
    )
    clean_dict = (
        clean_dict[["source", "target", "remove_occupation"]]
        .dropna()
        .reset_index(drop=True)
    )
    clean_dict["occupation_pair"] = clean_dict.apply(
        lambda x: [x["source"], x["target"]], axis=1
    )
    clean_dict = clean_dict[["occupation_pair", "remove_occupation"]].to_dict(
        orient="records"
    )

    data_merge = pd.merge(data, df_annotation, on="occupation")
    data_merge = data_merge.drop("occupation", axis=1).drop_duplicates()

    data_group = (
        data_merge.groupby("wikidata_id")["meta_occupation"].apply(list).reset_index()
    )
    data_group["meta_occupation"] = data_group["meta_occupation"].apply(
        lambda x: list(set(x))
    )
    data_group["meta_occupation"] = data_group["meta_occupation"].apply(
        lambda x: clean_list_occupation(clean_dict, x)
    )
    data_final = data_group.explode("meta_occupation")

    df_final_new = get_proportionned_occupations(
        data_final, data_merge, occupation_select="biologist"
    )
    df_final_new = get_proportionned_occupations(
        df_final_new, data_merge, occupation_select="naturalist"
    )
    df_final_new = df_final_new.drop_duplicates().reset_index(drop=True)

    replace_occupation = {
        "demographer": "geographer",
        "miitary specialist": "historian",
        "criminologist": "sociologist",
    }

    df_final_new["meta_occupation"] = df_final_new["meta_occupation"].apply(
        lambda x: replace_occupation.get(x, x)
    )

    conn = sqlite3.connect(DB_SCIENCE_PATH)
    data.to_sql("df_cleaned_occupations", conn, if_exists="replace", index=False)
