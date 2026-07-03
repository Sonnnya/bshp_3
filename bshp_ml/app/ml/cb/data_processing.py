import logging
import os
import pickle

import pandas as pd
from settings import USE_DETAILED_LOG
from sklearn.base import BaseEstimator, TransformerMixin

logging.getLogger("bshp_data_processing_logger")
logger = logging.getLogger(__name__)


class CBDataEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, parameters, y_col: str, name_col: str = None):
        self.parameters = parameters

        self.y = y_col
        self.name_col = name_col

        self.df = pd.DataFrame(columns=["name", "code1c", "code_norm"])

    def fit(self, X: pd.DataFrame, y=None):
        if USE_DETAILED_LOG:
            logger.info("Start encoding data, shape: %s", str(X.shape))

        X = X.copy()
        X[self.y] = X[self.y].astype(int)

        self.set_mapping(X)
        # self.set_txt_class_rate(X) #TODO: fix

        if USE_DETAILED_LOG:
            logger.info("Len of classes: %s", len(X[self.y].unique()))
        return self

    def set_mapping(self, X: pd.DataFrame) -> pd.DataFrame:
        y = self.y
        name = self.name_col
        if name == y:
            uniq = X[[y]].drop_duplicates(keep="first").reset_index(drop=True).copy()
            uniq["name"] = uniq[y]
            uniq[f"{y}_norm"] = uniq.index.astype(int)

            self.df = uniq.rename({y: "code1c", f"{y}_norm": "code_norm"}, axis=1)
        else:
            uniq = (X.drop_duplicates(subset=y, keep="first").copy())[
                [y, name]
            ].reset_index(drop=True)
            uniq[f"{y}_norm"] = uniq.index.astype(int)

            self.df = uniq.rename(
                {y: "code1c", name: "name", f"{y}_norm": "code_norm"}, axis=1
            )

        self.code2norm = dict(zip(self.df["code1c"].astype(int), self.df["code_norm"]))
        self.norm2code = dict(zip(self.df["code_norm"], self.df["code1c"].astype(int)))
        self.norm2name = dict(zip(self.df["code_norm"], self.df["name"]))
        self.code2name = dict(zip(self.df["code1c"], self.df["name"]))
        self.name2code = dict(zip(self.df["name"], self.df["code_norm"]))

        self.code2rate = None

    def set_txt_class_rate(self, X: pd.DataFrame):
        y = self.y
        name = self.name_col

        if f"pred_{name}" in X.columns and name in X.columns:
            # rate для класса: как часто именно его угадывает
            # текстовая модель (на обучающей выборке)
            self.code2rate = {
                code: len(
                    X[
                        (X[f"pred_{name}"] == X[f"{name}"])
                        & (X[name] == self.code2name[code])
                    ]
                )
                / len(X[X[f"{name}"] == self.code2name[code]])
                if len(X[X[f"{name}"] == self.code2name[code]]) != 0
                else 0
                for code in self.code2norm.keys()
            }
        else:
            self.code2rate = None

    def transform(self, X):
        X = X.copy()
        X[self.y] = X[self.y].replace(r"^\s*$", None, regex=True).fillna(-1).astype(int)

        if USE_DETAILED_LOG:
            logger.info("Encoding, shape: %s", str(X.shape))

        X[f"{self.y}_norm"] = X[self.y].map(self.code2norm).fillna(-1).astype(int)
        if USE_DETAILED_LOG:
            logger.info("Len of _norm classes: %s", len(X[f"{self.y}_norm"].unique()))
        if self.name_col:
            # if USE_DETAILED_LOG:
            #     logger.info(
            #         "Pre Encoded dict: %s, %s : %s (%s)",
            #         X[f"pred_{self.name_col}"].iloc[0],
            #         X[f"pred_pp_{self.name_col}"].iloc[0],
            #         list(self.name2code.keys())[0],
            #         list(self.name2code.values())[0],
            #     )
            #     logger.info(
            #         "Len of pred_name classes: %s %s",
            #         len(X[f"pred_{self.name_col}"].unique()),
            #         len(X[f"pred_pp_{self.name_col}"].unique()),
            #     )

            X[f"pred_{self.name_col}"] = X[f"pred_{self.name_col}"].replace(
                self.name2code
            )
            # X[f"pred_pp_{self.name_col}"] = X[f"pred_pp_{self.name_col}"].replace(
            #     self.name2code
            # )
            if self.code2rate is not None:
                X[f"class_rate_{self.name_col}"] = (
                    X[f"pred_{self.name_col}"]
                    .map(self.norm2code)
                    .map(self.code2rate)
                    .fillna(-1)
                )

            # if USE_DETAILED_LOG:
            #     logger.info(
            #         "Encoded dict: %s, %s, %s %s",
            #         X[f"pred_{self.name_col}"].iloc[0],
            #         X[f"pred_pp_{self.name_col}"].iloc[0],
            #         list(self.name2code.keys())[0],
            #         list(self.name2code.values())[0],
            #     )
            #     logger.info(
            #         "Len of pred_name classes: %s %s",
            #         len(X[f"pred_{self.name_col}"].unique()),
            #         len(X[f"pred_pp_{self.name_col}"].unique()),
            #     )
        X[self.y] = X[self.y].replace(r"^\s*$", None, regex=True).fillna(-1).astype(int)
        return X

    def inverse_transform(self, X):
        X = X.copy()
        if self.name_col != "year":
            # predicted name to corresponding label
            X[f"{self.name_col}"] = X[f"{self.y}_norm"].map(self.norm2name).fillna("")

            # X[self.y] = X[X[self.y] == -1][f"pred_{self.name_col}"].map(self.name2code)
            # X[self.name_col] = X[f"pred_{self.name_col}"]
        if USE_DETAILED_LOG:
            logger.info("Encoding, shape: %s", str(X.shape))
            logger.info(
                "Norm code: %s, 1CCode: %s, Name: %s",
                X[f"{self.y}_norm"].iloc[0],
                X[f"{self.y}_norm"].map(self.norm2code),
                X[f"{self.name_col}"].iloc[0],
            )
        X[self.y] = X[f"{self.y}_norm"].map(self.norm2code).fillna(-1)

        X[f"pred_{self.name_col}"] = (
            X[f"pred_{self.name_col}"].map(self.norm2name).fillna(-1)
        )

        return X

    # def _get_decoded_field(self, norm_value):
    #     if norm_value == -1:
    #         return ""
    #     else:
    #         return self.df[self.df[norm_value] == self.df["code_norm"]]["code1c"].iloc[
    #             0
    #         ]

    # def _get_encoded_field(self, value):
    #     if value not in self.df["code1c"]:
    #         return -1
    #     else:
    #         return self.df[self.df[value] == self.df["code1c"]]["code_norm"].iloc[0]

    # def _get_name(self, value):
    #     if not value:
    #         return ""
    #     else:
    #         return self.df[self.df[value] == self.df["code1c"]]["name"].iloc[0]

    # def _get_month(self, date_value: datetime):
    #     return date_value.month

    # def _get_year(self, date_value: datetime):
    #     return date_value.year

    def save(self, folder, name="encoder"):
        if not os.path.isdir(folder):
            os.makedirs(folder)
        with open(os.path.join(folder, f"{name}.pkl"), "wb") as fp:
            pickle.dump(self, fp)


def check_fields(
    df: pd.DataFrame,
    columns_to_check: list[str],
):
    mask = pd.Series(False, index=df.index)

    for col in columns_to_check:
        if col in df.columns:
            col_mask = (
                df[col].isna()  # NaN
                | (df[col].astype(str).str.strip() == "")
                | df[col].isin([0, -1, "0", "-1"])
            )
            mask = mask | col_mask

    if mask.any() and USE_DETAILED_LOG:
        logger.warning(f"Contains empty fields ({mask.sum()}):")
        logger.warning(
            "\n"
            + df.to_json(
                orient="records",
                force_ascii=False,
            )
        )
