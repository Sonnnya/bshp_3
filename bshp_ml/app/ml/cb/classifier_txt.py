import asyncio
import gc
import json
import logging
import os
import pickle
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
from catboost import CatboostError, Pool
from ml.cb.classifier import (
    CatBoostClassifier,
    sum_models,
)
from ml.data_processing import (
    Checker,
    FeatureAdder,
    NanProcessor,
)
from ml.metrics import MetricsTrain, write_in_csv
from schemas.models import ModelStatuses, ModelTypes
from settings import (
    DEVICES,
    MODEL_FOLDER,
    TASK_TYPE,
    THREAD_COUNT,
    USE_DETAILED_LOG,
    USED_RAM_LIMIT,
)
from sklearn.pipeline import Pipeline

from .classifier import CatBoostModel
from .data_processing import CBDataEncoder
from .utils import eval_model, make_all_data

logging.getLogger("bshp_data_processing_logger")
logger = logging.getLogger(__name__)
SEED = 42
ITEM = "cash_flow_item_code"
TEST_FRAC = 0.15


class CatBoostModelEmbeddings(CatBoostModel):
    """Класс модели catboost с пониманием эмбеддингов"""

    model_type = ModelTypes.catboost_txt

    field_models: dict[str, CatBoostClassifier] | None = None
    field_encoders: dict[str, CBDataEncoder] | None = None
    categorical: list[str] | None = (
        None  # названия категориальных колонок, need to be normalized
    )
    # cat_idxs: list[int] | None

    def __init__(self, base_name):
        super().__init__(base_name)
        self.need_to_encode = False
        self.field_encoders = {}
        self.categorical = [
            "moving_type",
            "company_inn",
            "base_document_kind",
            "contractor_inn",
            "contractor_kind",
            "cash_flow_item_code",
            "year",
            "cash_flow_details_code",
            "contract_name",  # TODO: number?
            "accepted_issued",
            "article_parent",
            # "article_group",
        ]
        self.fsttxt_columns = ["cash_flow_item_name", "cash_flow_details_name", "year"]
        self.float_columns.extend([f"prob_{y}" for y in self.fsttxt_columns])
        self.categorical.extend([f"pred_{y}" for y in self.fsttxt_columns])

        # self.categorical.extend([f"pred_pp_{y}" for y in self.fsttxt_columns])
        # self.float_columns.extend([f"prob_pp_{y}" for y in self.fsttxt_columns])
        # self.float_columns.extend([f"class_rate_{y}" for y in self.fsttxt_columns])

        self.str_columns.extend(
            [f"pred_{y}" for y in self.fsttxt_columns]
            + [
                "payment_purpose",
                "kind",
                "base_name",
                "base_document_number",
                "article_document_number",
                "article_code",
                "payment_purpose_returned",
                "contract_name",
                "contract_number",
                "accepted_issued",
            ]
        )
        self.x_columns.extend(
            [f"pred_{y}" for y in self.fsttxt_columns]
            # + [f"class_rate_{y}" for y in self.fsttxt_columns]
            + [
                "payment_purpose",
                "contract_name",
                "contract_number",
                "accepted_issued",
                "sin_month",
                "cos_month",
                "sin_day",
                "cos_day",
            ]
        )
        self.float_columns.extend(["sin_month", "cos_month", "sin_day", "cos_day"])
        self.columns_to_encode = self.categorical

        self.date_columns = [
            "date",
            "base_document_date",
            "article_document_date",
            "uploading_date",
        ]
        self.str_columns += self.date_columns
        self.parameters = {
            "x_columns": self.x_columns,
            "y_columns": self.y_columns,
            "str_columns": self.str_columns,
            "float_columns": self.float_columns,
            "bool_columns": self.bool_columns,
            "additional_columns": self.additional_columns,
            "columns_to_encode": self.columns_to_encode,
            "categorical": self.categorical,
            "date_columns": self.date_columns,
        }
        self.field_accuracies: dict[str, float] = {}

    def _fill_metrics(
        self,
        data_size: int,
        time_start: "datetime",
        time_end: "datetime",
    ) -> MetricsTrain:
        elapsed = (time_end - time_start).total_seconds()
        metrics = MetricsTrain(
            model_name=self.model_type.value,
            dataset_name=self.base_name,
            accuracy_year=self.field_accuracies.get("year", 0.0),
            accuracy_item=self.field_accuracies.get("cash_flow_item_code", 0.0),
            accuracy_details=self.field_accuracies.get("cash_flow_details_code", 0.0),
            time=elapsed,
            time_start=time_start.isoformat(),
            time_end=time_end.isoformat(),
            data_size=data_size,
        )
        write_in_csv(metrics)
        return metrics

    def train(
        self,
        params: dict,
        y: str,
        to_drop: list[str],
        cat_idxs: list[int],
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        test_pool: Pool,
        all_data: pd.DataFrame,
    ) -> CatBoostClassifier:
        """Trains one catboost model from scratch.
        <b>NOTE: before fitting add columns from fasttext model</b>"""
        model = None
        model_new = None

        BSIZE = 10_000
        for n, batch in enumerate(
            self.get_batch_pool(
                df=df_train,
                y=y,
                to_drop=to_drop,
                cat_idxs=cat_idxs,
                batch_size=BSIZE,
                all_data=all_data,
            )
        ):
            if USE_DETAILED_LOG:
                logger.info(f"BATCH {n + 1} of total {len(df_train) // BSIZE + 1}")
            if model is None:
                model = CatBoostClassifier(**params, allow_const_label=True)
                batch.set_baseline(np.zeros(shape=(batch.num_row(), len(all_data))))
                test_pool.set_baseline(
                    np.zeros(shape=(test_pool.num_row(), len(all_data)))
                )

                model.fit(X=batch, eval_set=test_pool)
            else:
                model_new = CatBoostClassifier(**params, allow_const_label=True)
                base = model.predict(batch, prediction_type="RawFormulaVal")
                base[np.isneginf(base)] = -1
                batch.set_baseline(base)

                test_base = model.predict(test_pool, prediction_type="RawFormulaVal")
                test_base[np.isneginf(test_base)] = -1
                test_pool.set_baseline(test_base)

                model_new.fit(X=batch, eval_set=test_pool)
                model = sum_models(
                    [model, model_new], ctr_merge_policy="IntersectingCountersAverage"
                )  # TODO: weights = [1/2] * 2?
                del model_new
                gc.collect()

        # model_new = CatBoostClassifier(**params)
        # test_base = model.predict(test_pool, prediction_type="RawFormulaVal")
        # test_base[np.isneginf(test_base)] = -1
        # test_pool.set_baseline(test_base)

        # model_new.fit(X=test_pool, eval_set=test_pool)
        # model = sum_models(
        #     [model, model_new], ctr_merge_policy="IntersectingCountersAverage"
        # )
        return model

    def gridsearch(
        self,
        all_classes: pd.DataFrame,
        df_train,
        df_test,
        cat_idxs: list[int],
        test_pool: Pool,
        y: str,
        to_drop: list[str],
        all_data: pd.DataFrame,
        lr=0.01,
        trees=20,
        _df_latest: None | pd.DataFrame = None,
    ):
        """
        Traines few models and choses best based on a given catboost parametrs
        """
        param_grid = {
            "learning_rate": [
                # lr,
                None,
            ],
            "depth": [6],
            "iterations": [
                # trees,
                # trees * 2,
                # max(int(trees * 0.7), 1),
                2**7,
                # 400,
            ],
            "l2_leaf_reg": [4],
        }

        best_score = 0
        best_model = None
        best_params = None

        # Перебираем все комбинации параметров
        for lr in param_grid["learning_rate"]:
            for depth in param_grid["depth"]:
                for iterations in param_grid["iterations"]:
                    if USE_DETAILED_LOG:
                        logging.info(
                            f"\n=== Testing: lr={lr}, depth={depth}, iterations={iterations} ==="
                        )

                    params = {
                        "task_type": TASK_TYPE,
                        "iterations": iterations,
                        "learning_rate": lr,
                        "depth": depth,
                        "classes_count": len(all_classes),
                        "early_stopping_rounds": int(iterations * 0.8) + 1,
                        "use_best_model": True,
                        "random_seed": SEED,
                        "verbose": USE_DETAILED_LOG * 100,
                        "loss_function": "MultiClass",
                        "devices": DEVICES,
                    }

                    # Обучаем модель
                    current_model = self.train(
                        params=params,
                        y=y,
                        to_drop=to_drop,
                        cat_idxs=cat_idxs,
                        df_train=df_train,
                        df_test=df_test,
                        test_pool=test_pool,
                        all_data=all_data,
                    )

                    # Оцениваем accuracy на тестовой выборке
                    preds = current_model.predict(test_pool, prediction_type="Class")

                    acc, f1 = eval_model(df_test[f"{y}"], preds)
                    tacc, _ = eval_model(
                        df_train[f"{y}"],
                        current_model.predict(df_train, prediction_type="Class"),
                    )

                    acc_after_test, _ = eval_model(df_test[f"{y}"], preds)
                    if USE_DETAILED_LOG:
                        logger.info(f"Accuracy: {acc_after_test:.4f}")
                    # Сохраняем лучшую модель
                    if acc > best_score:
                        best_score = acc

                        best_params = params
                        if USE_DETAILED_LOG:
                            logger.info(
                                f"*** New best model! Test Accuracy: {acc:.4f} ***, ***train accuracy: {tacc:.4f}*** Best params: {best_params}"
                            )

                        if USE_DETAILED_LOG:
                            logger.info(f"Accuracy: {acc:.4f}")
                            logger.info("Learn on test data...")

                        if USE_DETAILED_LOG:
                            logger.info(
                                f"Model loss_function: {current_model.get_params().get('loss_function')}"
                            )
                            logger.info(
                                f"Model classes_count: {current_model.get_params().get('classes_count')}"
                            )
                            logger.info(
                                f"Baseline shape: {test_pool.get_baseline().shape if test_pool.get_baseline() is not None else None}"
                            )

                        if _df_latest is not None and len(_df_latest) > 0:
                            final_batch = pd.concat(
                                [df_test, _df_latest], ignore_index=True
                            )
                        else:
                            final_batch = df_test

                        # Дообучить для прода
                        _params = params.copy()
                        _params["use_best_model"] = False

                        current_model_test = CatBoostClassifier(
                            **_params, allow_const_label=True
                        )
                        current_model_test = current_model_test.fit(
                            X=final_batch.drop(y, axis=1),
                            y=final_batch[y],
                            init_model=current_model,
                            cat_features=cat_idxs,
                        )
                        best_model = current_model_test
        return best_model, best_score

    def train_on_field(
        self, df: pd.DataFrame, y: str, to_drop: list[str], parameters: dict
    ):
        """train model on a given field y"""
        if USE_DETAILED_LOG:
            logger.info(f"Predict {len(df[y])} records")

        bscores = []

        df = df.copy()
        # TODO: fsttxt class feature
        # TODO: article_parent rm numbers
        # df["article_parent"] = (
        #     df["article_parent"]
        #     .str.replace(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", regex=True)
        #     .str.strip()
        # )
        # df["article_group"] = (
        #     df["article_group"]
        #     .str.replace(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", regex=True)
        #     .str.strip()
        # )

        UNFEATURED = (
            [
                "company_inn",
                "contractor_name",
                "contractor_kpp",
                "qty",
                "price",
                "sum",
                "contractor_account_number",
                "company_account_number",
                "article_row_number",
                "row_number",
                "number",
            ]
            + [f"pred_pp_{y}" for y in self.fsttxt_columns]
            + [
                f"class_rate_{y}" for y in self.fsttxt_columns
            ]  # wrong count somitimes. TODO: fix counting that field in encoder and check accuracies
        )

        if y != "year":
            UNFEATURED += ["pred_pp_year", "prob_pp_year", "pred_year", "prob_year"]

        categorical = [col for col in self.categorical if col not in UNFEATURED]

        _df_latest = (
            df.sort_values(
                by=["document_year", "document_month"], ascending=[True, True]
            )
            .tail(10_000)
            .copy()
        )
        df = df.drop(
            labels=[
                col
                for col in UNFEATURED
                if col not in categorical and col in df.columns
            ],
            axis=1,
        )
        _df_latest = _df_latest.drop(
            labels=[
                col
                for col in UNFEATURED
                if col not in categorical and col in df.columns
            ],
            axis=1,
        )
        _df_latest = _df_latest[df.columns]
        # drop here? No ' '(-1) for models?

        if y == "cash_flow_details_code":
            self.field_models[y] = {}
            self.field_encoders[y] = {}
            for item in df["cash_flow_item_code"].unique():
                # model = None
                if USE_DETAILED_LOG:
                    logger.info(
                        f"Item {item}: found {len(df[df['cash_flow_item_code'] == item])} records"
                    )
                # group and get df_i sample
                df_i = df[df["cash_flow_item_code"] == item].copy()

                # det_map1 = {name: num for num, name in enumerate(df_i[y].unique())}
                # det_unmap1 = {v: k for k, v in det_map1.items()}

                # df_i[f"{y}_norm"] = df_i[y].copy().map(det_map1)

                encoder = CBDataEncoder(
                    parameters=self.parameters,
                    y_col=y,
                    name_col="cash_flow_details_name",
                )
                df_i = encoder.fit_transform(df_i, None)
                encoder.save(os.path.join(MODEL_FOLDER, self.uid, y), int(item))
                self.field_encoders[y][int(item)] = encoder

                to_drop = [y for y in to_drop if y in df_i.columns]
                to_drop.extend(
                    [
                        c
                        for c in self.str_columns
                        + self.fsttxt_columns
                        + ["uploading_date"]
                        if (c in df_i.columns and c not in categorical)
                    ]
                )
                to_drop.extend(
                    [
                        c
                        for c in UNFEATURED
                        if c not in categorical and c in df_i.columns
                    ]
                )

                df_i.drop(to_drop, inplace=True, axis=1)

                # Guard: constant features — no model can learn, use majority class
                _feature_cols = [c for c in df_i.columns if c != f"{y}_norm"]
                if not _feature_cols or df_i[_feature_cols].nunique().max() <= 1:
                    this_label = df_i[f"{y}_norm"].mode().iloc[0]
                    if USE_DETAILED_LOG:
                        logger.warning(
                            "Item %s has constant features after column drop — using majority label %s",
                            item,
                            this_label,
                        )
                    self.field_accuracies.setdefault(y, []).append(0.0)
                    self.strict_acc[y][int(item)] = this_label
                    continue

                all_data = make_all_data(df_i, f"{y}_norm")
                all_classes = all_data[f"{y}_norm"].unique()

                if USE_DETAILED_LOG:
                    logger.info("Total len of classes data: %s", len(all_data))
                    logger.info(f"X columns: {df_i.columns}, shape: {df_i.shape}")
                if len(all_data) == 1:
                    self.field_accuracies.setdefault(y, []).append(1.0)
                    # this_label = det_unmap1.get(all_data[f"{y}_norm"].iloc[0])
                    this_label = all_data[f"{y}_norm"].iloc[0]
                    bscores.extend([1 for _ in range(len(df_i[f"{y}_norm"]))])
                    if USE_DETAILED_LOG:
                        logger.info("Only one details code, no need for model")
                    # TODO: сохранить словарь
                    self.strict_acc[y][int(item)] = this_label
                    continue

                # X, y train
                df_i = df_i.query(f"`{y}_norm` not in ['', ' '] and `{y}_norm` != -1")
                all_data = make_all_data(df_i, f"{y}_norm")
                all_classes = all_data[f"{y}_norm"].unique()

                df_test = df_i.sample(
                    frac=TEST_FRAC, random_state=SEED
                )  # might need better split, data is strongly time-dependent

                _df_latest_i = encoder.transform(_df_latest)
                _df_latest_i.drop(to_drop, inplace=True, axis=1)
                _df_latest_i = _df_latest_i.query(
                    f"`{y}_norm` not in ['', ' '] and `{y}_norm` != -1"
                )

                df_train = df_i.drop(df_test.index)

                if len(df_test) == 0:
                    df_test = df_train.copy()

                df_test = self.make_full(
                    df_test, all_data, f"{y}_norm", 0, len(df_test)
                )
                df_train = self.make_full(
                    df_train, all_data, f"{y}_norm", 0, len(df_train)
                )
                # cats indxs
                cat_idxs = [
                    df_i.columns.get_loc(key=cat)
                    for cat in categorical
                    if cat in df_i.columns
                ]

                if USE_DETAILED_LOG:
                    #     logger.info("Total len of classes data: %s", len(all_data))
                    if df_i.isnull().any().any() or df_test.isnull().any().any():
                        logger.warning(
                            f"None columns detected: {df_i.columns[df_i.isnull().any()]}, {df_test.columns[df_test.isnull().any()]}"
                        )

                test_pool = self.get_test_pool(
                    df_test=df_test,
                    to_drop=[],
                    cat_idxs=cat_idxs,
                    all_data=all_data,
                    y=f"{y}_norm",
                )
                try:
                    model_i, acc_i = self.gridsearch(
                        all_classes=all_classes,
                        df_train=df_train,
                        df_test=df_test,
                        cat_idxs=cat_idxs,
                        test_pool=test_pool,
                        y=f"{y}_norm",
                        to_drop=[],
                        lr=parameters.get("lr", 0.01),
                        trees=parameters.get("trees", 30),
                        all_data=all_data,
                        _df_latest=_df_latest_i,
                    )
                    self.field_models[y][int(item)] = model_i
                    self.field_accuracies.setdefault(y, []).append(acc_i)
                except CatboostError as e:
                    logger.error(f"Error while training model for item {item}: {e}")
                    self.field_accuracies.setdefault(y, []).append(0.0)
                    raise e

                # TODO: если fsttxt model справляется лучше?
                if False:
                    if (
                        len(
                            df[
                                df["pred_cash_flow_item_name"]
                                == df["cash_flow_item_name"]
                            ]
                        )
                        / len(df)
                        > ...
                    ):
                        self._save_cb_model(model_i, column=y, item=int(item))
                        ...
                self._save_cb_model(model_i, column=y, item=int(item))
            # AVG for details
            if isinstance(self.field_accuracies.get(y), list):
                self.field_accuracies[y] = (
                    float(sum(self.field_accuracies[y]) / len(self.field_accuracies[y]))
                    if self.field_accuracies[y]
                    else 0.0
                )

        else:
            if USE_DETAILED_LOG:
                logger.info(f"Found {len(df[y])} records")

            encoder = CBDataEncoder(
                parameters=self.parameters,
                y_col=y,
                name_col="cash_flow_item_name"
                if y == "cash_flow_item_code"
                else "year",
            )  # encoder adds some fields too ... TODO: one place for all
            df = encoder.fit_transform(df, None)
            encoder.save(os.path.join(MODEL_FOLDER, self.uid, y))
            self.field_encoders[y] = encoder

            to_drop = [y for y in to_drop if y in df.columns]
            to_drop.extend(
                [
                    c
                    for c in self.str_columns + self.fsttxt_columns + ["uploading_date"]
                    if (c in df.columns and c not in self.categorical)
                ]
            )
            to_drop.extend(
                [c for c in UNFEATURED if c not in categorical and c in df.columns]
            )
            df.drop(to_drop, axis=1, inplace=True)

            _feature_cols = [c for c in df.columns if c != f"{y}_norm"]
            if not _feature_cols or df[_feature_cols].nunique().max() <= 1:
                this_label = df[f"{y}_norm"].mode().iloc[0]
                if USE_DETAILED_LOG:
                    logger.warning(
                        "Field %s has constant features after column drop — using majority label %s",
                        y,
                        this_label,
                    )
                self.field_accuracies[y] = 0.0
                self.strict_acc[y] = this_label
                return

            all_data = make_all_data(df, f"{y}_norm")
            all_classes = all_data[f"{y}_norm"].unique()
            if USE_DETAILED_LOG:
                logger.info(
                    "Total len of classes data: %s, len of all_classes: %s",
                    len(all_data),
                    len(all_classes),
                )
                logger.info(f"X columns: {df.columns}, shape: {df.shape}")

            if len(all_data) == 1:
                self.field_accuracies.setdefault(y, []).append(1.0)
                # TODO: get rid of this, save encoder to .pkl
                # this_label = det_unmap1.get(all_data[f"{y}_norm"].iloc[0])
                this_label = all_data[f"{y}_norm"].iloc[0]
                bscores.extend([1 for _ in range(len(df[f"{y}_norm"]))])
                self.strict_acc[y] = this_label
                return

            # X, y train
            df = df.query(f"`{y}_norm` not in ['', ' '] and `{y}_norm` != -1")
            all_data = make_all_data(df, f"{y}_norm")
            all_classes = all_data[f"{y}_norm"].unique()

            df_test = df.sample(frac=TEST_FRAC, random_state=SEED)

            _df_latest = encoder.transform(_df_latest)
            _df_latest.drop(to_drop, inplace=True, axis=1)
            _df_latest = _df_latest.query(
                f"`{y}_norm` not in ['', ' '] and `{y}_norm` != -1"
            )

            df_train = df.drop(df_test.index)

            cat_idxs = [
                df.columns.get_loc(key=cat)
                for cat in self.categorical
                if cat in df.columns
            ]
            if len(df_test) == 0:
                df_test = df_train.copy()

            df_test = self.make_full(df_test, all_data, f"{y}_norm", 0, len(df_test))
            df_train = self.make_full(df_train, all_data, f"{y}_norm", 0, len(df_train))

            test_pool = self.get_test_pool(
                df_test=df_test,
                to_drop=[],
                y=f"{y}_norm",
                cat_idxs=[
                    df_test.columns.get_loc(key=cat)
                    for cat in self.categorical
                    if cat in df_test.columns
                ],
                all_data=all_data,
            )
            model_i, acc_i = self.gridsearch(
                all_classes=all_classes,
                df_train=df_train,
                df_test=df_test,
                cat_idxs=cat_idxs,
                test_pool=test_pool,
                y=f"{y}_norm",
                to_drop=[],
                lr=parameters.get("lr", 0.01),
                trees=parameters.get("trees", 30),
                all_data=all_data,
                _df_latest=None if y == "year" else _df_latest,
            )
            self.field_models[y] = model_i
            self.field_accuracies[y] = float(acc_i)
            self._save_cb_model(model_i, y)

    async def _fit(self, df: pd.DataFrame, parameters: dict, is_first=True):
        return await asyncio.to_thread(self.__sync_fit, df, parameters, is_first)

    def __sync_fit(self, df: pd.DataFrame, parameters: dict, is_first=True):
        is_first = True  # TODO: ?
        self.field_accuracies = {}
        time_start = datetime.utcnow()
        if USE_DETAILED_LOG:
            logger.info("{} fit".format("First" if is_first else "continuous"))
            logger.info(f"Starting time: {time_start}, Shape: {df.shape}")

        for y in self.y_columns:
            self.strict_acc[y] = {}
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y))
            self.train_on_field(
                df=df, y=y, to_drop=self.y_columns, parameters=parameters
            )
            gc.collect()

        time_end = datetime.utcnow()
        self._fill_metrics(
            data_size=len(df),
            time_start=time_start,
            time_end=time_end,
        )
        # self._save_cb_model(model, column=y, item=item)
        # self._load_all_models()
        # self._load_all_models()

    async def fit(self, Xy_api: dict, parameters):
        X_y = pd.DataFrame(Xy_api)
        logger.info("Fitting")
        try:
            self.field_encoders = {}
            self.field_models = {}
            self.items_wo_year = None
            need_to_initialize = (
                self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR]
                or parameters.get("refit") == 0
            )
            calculate_metrics = parameters.get("calculate_metrics")
            use_cross_validation = parameters.get("use_cross_validation")

            await self._before_fit(
                parameters, need_to_initialize, calculate_metrics, use_cross_validation
            )

            train_test_indexes = None
            self.metrics_dataset_name = ""
            self.test_metrics_dataset_name = ""
            if use_cross_validation:
                # irrelevant, in external api layer we always use = False
                # cross validation is used in train_on_field instead, every time anyway, not accounting this parameter
                if USE_DETAILED_LOG:
                    logger.info(
                        "Calculating train/test indexes for metrics (use_cross_validation=True)"
                    )
                train_test_indexes = self._get_train_test_indexes(X_y)
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(
                        X_y.iloc[train_test_indexes[0]]
                    )
                    self.test_metrics_dataset_name = await self._save_dataset_to_temp(
                        X_y.iloc[train_test_indexes[1]]
                    )
            else:
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(X_y)

            datasets = await self._transform_dataset(
                X_y,
                parameters,
                need_to_initialize,
                train_test_indexes,
                calculate_metrics,
            )

            await self._fit(datasets["train"], parameters, is_first=need_to_initialize)

            if calculate_metrics:
                await self._calculate_metrics(
                    parameters, need_to_initialize, use_cross_validation
                )

            await self._after_fit(
                parameters,
                need_to_initialize=need_to_initialize,
                use_cross_validation=use_cross_validation,
            )

        except Exception as e:
            await self._on_fitting_error(e)

    def __save(self, model: CatBoostClassifier, path: str):
        model.save_model(path)

    async def predict(self, X: pd.DataFrame, for_metrics=False):
        return await asyncio.to_thread(self._sync_predict, X, for_metrics)

    def _sync_predict(self, X: pd.DataFrame, for_metrics=False):
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError("Model is not ready. Fit it before.")
        # load for details code
        field_models = self.field_models
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")
        X = pd.DataFrame(X)

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters, for_predict=True)))
        pipeline_list.append(
            ("nan_processor", NanProcessor(self.parameters, for_predict=True))
        )
        pipeline_list.append(
            ("feature_addder", FeatureAdder(self.parameters, for_predict=True))
        )
        # TODO: do i load fitted?
        # if self.need_to_encode:
        #     pipeline_list.append(("data_encoder", self.data_encoder))

        pipeline = Pipeline(pipeline_list)

        for y in self.y_columns:
            X[y] = ""
        # set_config(transform_output="pandas")
        # X["article_parent"] = (
        #     X["article_parent"]
        #     .str.replace(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", regex=True)
        #     .str.strip()
        # )
        # X["article_group"] = (
        #     X["article_group"]
        #     .str.replace(r"[^a-zA-Zа-яА-ЯёЁ\s]", " ", regex=True)
        #     .str.strip()
        # )
        X_y = pipeline.fit_transform(X).copy()
        # c_x_columns = self.x_columns + [
        #     "number",
        #     "date",
        #     # "pred_cash_flow_item_name",
        #     # "pred_cash_flow_details_name",
        # ]
        # TODO: не надо предсказывать пустые, если используем модели?
        # check_fields( # just logs (too much)
        #     X_y,
        #     [
        #         col
        #         for col in set(self.x_columns) | set(self.str_columns)
        #         if col in X_y.columns
        #     ],
        # )

        for y in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start predicting. Field = "{}"'.format(y))
            if y == "cash_flow_details_code":
                cash_flow_items = list(X_y["cash_flow_item_code"].unique().astype(int))

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:
                        logger.info("Predicting {} - {}".format(ind, item_col))

                    Xy1 = X_y.loc[X_y["cash_flow_item_code"] == item_col].copy()

                    encoder = self.field_encoders[y].get(int(item_col))
                    if encoder is None:
                        continue
                    Xy1 = encoder.transform(Xy1)

                    if (
                        self.strict_acc.get(y) is not None
                        and self.strict_acc[y].get(int(item_col)) is not None
                    ):
                        # No model nedeed
                        Xy1[f"{y}_norm"] = self.strict_acc[y][int(item_col)]
                    else:
                        # TODO: поправить, чтобы везде были либо инты, либо строки с 0ми
                        # encoder = self.field_encoders[y].get(int(item_col))
                        # if encoder is None:
                        #     continue
                        # Xy1 = encoder.transform(Xy1)
                        # model = field_models[y][int(item_col)]

                        model = field_models[y][int(item_col)]
                        Xy1 = Xy1[[feat for feat in model.feature_names_]]
                        logger.info(
                            "Features: %s",
                            [(i, feat) for i, feat in enumerate(model.feature_names_)],
                        )
                        cat_idxs = [
                            Xy1.columns.get_loc(key=cat)
                            for cat in self.categorical
                            if cat in Xy1.columns
                        ]

                        X_pool = Pool(
                            Xy1,
                            cat_features=cat_idxs,
                        )

                        y_pred = model.predict(X_pool, prediction_type="Class")

                        Xy1[f"{y}_norm"] = y_pred.ravel()

                        if USE_DETAILED_LOG:
                            logger.info(f"Feature names: {model.feature_names_}")
                            importance = (
                                pd.DataFrame(
                                    {
                                        "imp": model.get_feature_importance(),
                                        "names": model.feature_names_,
                                    }
                                )
                                .sort_values("imp", ascending=False)
                                .head()
                            )
                            logger.info(
                                f"For {y} most important fields are:\n%s",
                                importance.to_json,
                            )
                    Xy1 = encoder.inverse_transform(Xy1)
                    # X_y[X_y["cash_flow_item_code"] == item_col][y] = Xy1[y]

                    mask = X_y["cash_flow_item_code"] == item_col
                    X_y.loc[mask, Xy1.columns] = Xy1.values
                    # X_y[X_y["cash_flow_item_code"] == item_col][y] = Xy1[y]
            else:
                # c_y_columns = [y]
                if (
                    self.strict_acc.get(y) is not None
                    and len(self.strict_acc.get(y)) != 0
                ):
                    X_y[y] = self.strict_acc[y]
                else:
                    # if y == "year":
                    #     year_mask = X_y[ITEM].isin(self.items_wo_year)
                    X = X_y.copy()
                    model = field_models[y]
                    if USE_DETAILED_LOG:
                        logger.info("Encoders: %s", list(self.field_encoders.keys()))
                    encoder = self.field_encoders[y]
                    X = encoder.transform(X)
                    # if "pred_cash_flow_item_name" in X.columns:
                    # ymap = {
                    #     x: i
                    #     for i, x in enumerate(
                    #         X["pred_cash_flow_item_name"].unique()
                    #     )
                    # }
                    # # TODO: это в декодер
                    # X[f"{y}_norm"] = X["pred_cash_flow_item_name"].map(ymap)
                    # to_drop = self.y_columns
                    # else:
                    #     logging.error("No results for {y} from fasttext")
                    X = X[[feat for feat in model.feature_names_]]
                    cat_idxs = [
                        X.columns.get_loc(key=cat)
                        for cat in self.categorical
                        if cat in X.columns
                    ]

                    X_pool = Pool(
                        X,
                        cat_features=cat_idxs,
                    )
                    if USE_DETAILED_LOG:
                        logger.info(f"Feature names: {model.feature_names_}")
                        importance = (
                            pd.DataFrame(
                                {
                                    "imp": model.get_feature_importance(),
                                    "names": model.feature_names_,
                                }
                            )
                            .sort_values("imp", ascending=False)
                            .head()
                        )
                        logger.info(
                            f"For {y} most important fields are:\n%s",
                            importance.to_json,
                        )

                    predictions = model.predict(X_pool, prediction_type="Class")

                    X_y[f"{y}_norm"] = predictions.ravel()
                    X_y = encoder.inverse_transform(X_y)
                    # if y == "year":
                    #     X_y.loc[year_mask, y] = ""

                    if USE_DETAILED_LOG:
                        _js = json.loads(
                            X_y.head().to_json(
                                orient="records",
                                force_ascii=False,
                            )
                        )
                        logger.info(
                            'Predictions ITEM: \n"{}"'.format(
                                json.dumps(_js, indent=4, ensure_ascii=False)
                            )
                        )

                if USE_DETAILED_LOG:
                    logger.info('Predicting model. Field = "{}". Done'.format(y))
                    logger.info("Predicted: %s", X_y[y])
        for y in self.y_columns:
            if y != "year":
                X_y[y] = X_y[y].astype(str).str.zfill(9)
            else:
                X_y[y] = X_y[y].astype(str)
        return X_y.to_dict(orient="records")

        # records = X_y.to_dict(orient="records")
        # # Sanitize non-finite floats before serialization. The per-item details
        # # loop writes back inconsistent column sets across items (e.g. strict
        # # items keep the encoder-created class_rate_*, model items drop it; unseen
        # # items leave *_norm unfilled), so internal helper columns can end up NaN.
        # # Starlette's JSONResponse uses allow_nan=False and 500s on NaN/inf.
        # for record in records:
        #     for key, value in record.items():
        #         if isinstance(value, float) and not math.isfinite(value):
        #             record[key] = None
        # return records
        # if self.need_to_encode:
        #     X_y = pipeline.named_steps["data_encoder"].inverse_transform(X_y)

        # for y in self.y_columns:
        #     X_result[y] = X_y[y]

        # if for_metrics:
        #     return X_result
        # else:
        #     return X_result.to_dict(orient="records")

    def _get_cb_model(self, parameters):
        epochs = parameters.get("epochs", 20)
        depth = parameters.get("depth", 8)
        model = CatBoostClassifier(
            iterations=epochs,
            learning_rate=0.1,
            depth=depth,
            thread_count=THREAD_COUNT,
            used_ram_limit=USED_RAM_LIMIT,
        )

        return model

    def make_full(
        self, df: pd.DataFrame, all_data: pd.DataFrame, y: str, i=0, j=None
    ) -> pd.DataFrame:
        """
        fills dataset for each batch (df[i:j]) to have every class presented in all_classes
        """
        if j is None:
            j = len(df)

        all_data = all_data.copy().reset_index(drop=True)

        # classes_df = df[i:j].groupby(df[y]).first().copy()
        classes_df = df[i:j].copy().drop_duplicates(subset=[y], keep="first")
        # classes_df.reset_index(drop=True, inplace=True)

        missing = all_data[~all_data[y].isin(classes_df[y])]
        if not missing.empty:
            Xy = pd.concat(
                [
                    df[i:j],
                    missing,
                ],
                ignore_index=True,
            )
        else:
            Xy = df[i:j]
        return Xy

    def get_batch_pool(
        self,
        df: pd.DataFrame,
        y: str,
        all_data: pd.DataFrame,
        to_drop: list[str],
        cat_idxs: list[int],
        batch_size: int = 1000,
    ):
        """Batches Pool generator"""
        i = 0
        j = 0
        while i < len(df):
            j = min(i + batch_size, len(df))
            Xy = self.make_full(df=df, all_data=all_data, y=f"{y}", i=i, j=j)
            y_batch = Xy[f"{y}"]
            # X_batch = Xy.drop([y for y in self.y_columns if y in Xy.columns], axis=1) #old thing, overwritten anyway
            # TODO: columns to upper layer
            X_batch = Xy.drop(columns=set(to_drop) | {y}, axis=1)
            # if X_batch has constant columns (all same)
            if X_batch.nunique().max() == 1:
                if USE_DETAILED_LOG:
                    logger.warning(
                        "Batch has constant columns for field: %s, Values: %s",
                        y,
                        y_batch,
                    )
                y_batch[:] = y_batch.iloc[0]
            pool = Pool(
                X_batch,
                label=y_batch,
                cat_features=cat_idxs,
            )
            # if QUANTIZE:
            #     pool.quantize()
            yield pool
            del Xy, y_batch, X_batch
            gc.collect()
            i = j

    def get_test_pool(
        self,
        df_test: pd.DataFrame,
        y: str,
        all_data,
        to_drop: list[str],
        cat_idxs: list[int],
    ):
        # self.field_models[y]
        self.make_full(df=df_test, y=y, all_data=all_data, i=0, j=len(df_test))
        y_test = df_test[f"{y}"]
        # X_test = df_test.drop([y for y in self.y_columns if y in df_test.columns], axis=1)
        X_test = df_test.drop(columns=set(to_drop) | {y}, axis=1)
        X_test.columns
        # text_features=txts)
        pool = Pool(
            X_test,
            label=y_test,
            cat_features=cat_idxs,
        )
        # if QUANTIZE:
        #     pool.quantize()

        return pool

    def _load_cb_model(self, column, item=None, number=None) -> CatBoostClassifier:
        path_to_model_folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if item is not None:
            path_to_model = os.path.join(path_to_model_folder, "{}.cbm".format(item))
            encoder = self._load_cb_encoder(path_to_model_folder, f"{item}.pkl")
        elif number is not None:
            path_to_model = os.path.join(
                path_to_model_folder, "{}.cbm".format(str(number))
            )
            encoder = self._load_cb_encoder(path_to_model_folder, f"{number}.pkl")
        else:
            path_to_model = os.path.join(path_to_model_folder, "sum.cbm")
            encoder = self._load_cb_encoder(path_to_model_folder)

        if not os.path.isdir(path_to_model_folder):
            os.makedirs(path_to_model_folder)
        model = CatBoostClassifier()
        model.load_model(path_to_model)

        if item is not None:
            if not self.field_models.get(column):
                self.field_models[column] = {}
            self.field_models[column][int(item)] = model
            self.field_encoders[column][int(item)] = encoder
            logging.info("Encoder loaded: %s %s", column, str(item))
        else:
            self.field_encoders[column] = encoder
            self.field_models[column] = model
            logging.info("Encoder loaded: %s", column)

    def _load_column_models(self, column, sum_model=False):
        folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filenames = os.listdir(folder)
        models = {}
        encoders = {}

        for filename in filenames:
            if filename == "sum.cbm" and not sum_model:
                continue
            if filename != "sum.cbm" and sum_model:
                continue
            logging.info("Loading %s", filename)
            item = filename.split(".")[0]
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODEL_FOLDER, self.uid, column, filename))
            models[item] = model
            encoders[item] = self._load_cb_encoder(
                os.path.join(MODEL_FOLDER, self.uid, column),
                "encoder.pkl" if filename == "sum.cbm" else filename,
            )

        return models, encoders

    def _load_column_model(self, column, item=None):
        col_path = os.path.join(MODEL_FOLDER, self.uid, column)

        if item is not None:
            item = int(item)
            pkl_path = os.path.join(col_path, f"{item}.pkl")
            json_path = os.path.join(col_path, f"{item}.json")
            cbm_path = os.path.join(col_path, f"{item}.cbm")

            if not os.path.exists(pkl_path) and USE_DETAILED_LOG:
                logger.warning(
                    "No encoder file %s for item %s in column %s — skipping",
                    pkl_path,
                    item,
                    column,
                )
                return

            if not self.field_encoders.get(column):
                self.field_encoders[column] = {}
            self.field_encoders[column][item] = self._load_cb_encoder(
                col_path, f"{item}.pkl"
            )

            if os.path.exists(json_path):
                if not self.strict_acc.get(column):
                    self.strict_acc[column] = {}

                with open(json_path) as fp:
                    self.strict_acc[column][int(item)] = np.int64(
                        json.load(fp)["value"]
                    )
            elif os.path.exists(cbm_path):
                self._load_cb_model(column, int(item))
        else:
            if not self.field_encoders.get(column):
                self.field_encoders[column] = {}

            self.field_encoders[column] = self._load_cb_encoder(col_path)

            if os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, "sum.json")):
                with open(
                    os.path.join(MODEL_FOLDER, self.uid, column, "sum.json")
                ) as fp:
                    self.strict_acc[column] = np.int64(json.load(fp)["value"])
            elif os.path.exists(
                os.path.join(MODEL_FOLDER, self.uid, column, "sum.cbm")
            ):
                self._load_cb_model(column)

    def _load_all_models(self):
        for y in self.y_columns:
            models, encoders = self._load_column_models(
                y, sum_model=y != "cash_flow_details_code"
            )
            logger.info("Encoders: %s", encoders)
            if models:
                if y != "cash_flow_details_code":
                    self.field_models[y] = list(models.values())[0]
                    self.field_encoders[y] = list(encoders.values())[0]
                else:
                    for item, model in models.items():
                        self.field_models[y][int(item)] = model
                        self.field_encoders[y][int(item)] = encoders[item]

    def _delete_submodels(self, column):
        path_to_dir = os.path.join(MODEL_FOLDER, self.uid, column)

        filenames = os.listdir(path_to_dir)
        for filename in filenames:
            if filename == "sum.cbm":
                continue

            os.remove(os.path.join(path_to_dir, filename))

    async def _transform_dataset(
        self,
        dataset,
        parameters,
        need_to_initialize,
        train_test_indexes=None,
        calculate_metrics=False,
    ):
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        pipeline_list = []
        pipeline_list.append(("checker", Checker(self.parameters)))
        pipeline_list.append(("nan_processor", NanProcessor(self.parameters)))
        pipeline_list.append(("feature_adder", FeatureAdder(self.parameters)))
        # old logic
        # if self.need_to_encode:
        #     if need_to_initialize:
        #         self.data_encoder = DataEncoder(self.parameters)
        #     self.data_encoder.form_encode_dict = need_to_initialize
        #     pipeline_list.append(("data_encoder", self.data_encoder))

        # pipeline_list.append(("shuffler", Shuffler(self.parameters)))

        pipeline = Pipeline(pipeline_list)
        dataset = pipeline.fit_transform(dataset)
        # this shuffle instead of sklearn.shuffler
        dataset = dataset.sample(frac=1, random_state=SEED).reset_index(drop=True)
        if USE_DETAILED_LOG:
            logger.info("Performed shuffle, shape: %s", dataset.shape)

        for y in self.y_columns:
            dataset[y] = dataset[y].replace(r"^\s*$", -1, regex=True)

        datasets = {}
        if train_test_indexes:
            datasets["train"] = dataset.iloc[train_test_indexes[0]]
            datasets["test"] = dataset.iloc[train_test_indexes[1]]
        else:
            datasets["train"] = dataset

        # Collect classes from the train split so saved classes match trained items.
        # Using the full dataset caused missing encoder files when items appeared only
        # in the test split (cross-validation), breaking model load later.
        train = datasets["train"]
        self.classes = {}
        for y in self.y_columns:
            if y == "cash_flow_details_code":
                self.classes[y] = {}
                for item in dataset["cash_flow_item_code"].unique():
                    self.classes[y][int(item)] = list(
                        dataset[dataset["cash_flow_item_code"] == int(item)][y]
                        .unique()
                        .astype(int)
                    )
            else:
                self.classes[y] = list(dataset[y].unique().astype(int))

        gc.collect()
        return datasets

    async def _on_fitting_error(self, ex):
        self._delete_all_models()
        await super()._on_fitting_error(ex)

    # def _save_encoder(self):
    #     path_to_model = os.path.join(MODEL_FOLDER, self.uid)
    #     if not os.path.isdir(path_to_model):
    #         os.makedirs(path_to_model)

    #     with open(os.path.join(path_to_model, "encoder.pkl"), "wb") as fp:
    #         pickle.dump(self.data_encoder, fp)
    async def save(self, without_models=False):
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        path_to_model = os.path.join(MODEL_FOLDER, self.uid)
        if os.path.isdir(path_to_model):
            shutil.rmtree(path_to_model)

        if USE_DETAILED_LOG:
            logger.info("Saving models in %s", os.path.join(MODEL_FOLDER, self.uid))

        if not without_models:
            for y_col in self.y_columns:
                path_to_col = os.path.join(MODEL_FOLDER, self.uid, y_col)
                if not os.path.isdir(path_to_col):
                    os.makedirs(path_to_col)
                if y_col == "cash_flow_details_code":
                    for item in self.classes[y_col]:
                        self._save_column_model(y_col, int(item))
                        self._save_field_encoder(y_col, int(item))
                else:
                    self._save_column_model(y_col)
                    self._save_field_encoder(y_col)
        self._save_parameters()

    def _save_field_encoder(self, column, item=None):
        if not os.path.isdir(os.path.join(MODEL_FOLDER, self.uid, column)):
            os.makedirs(os.path.join(MODEL_FOLDER, self.uid, column))
        col_path = os.path.join(MODEL_FOLDER, self.uid, column)

        if item is not None:
            item = int(item)
            if item not in self.field_encoders[column]:
                logging.warning(
                    f"No item {item} in {list(self.field_encoders.keys())}. No encoder to save"
                )
            else:
                encoder = self.field_encoders[column][item]
                encoder.save(col_path, item)
            # logging.warning(
            #     f"No item {item} in {[v[item].df for k, v in self.field_encoders.items() if isinstance(v, dict)]}"
            # )
            # TODO: fix если не сбрасывали обучения ПОСЛЕ ОШИБОЧНОГО, то ошибка
        elif self.field_encoders.get(column):
            encoder = self.field_encoders[column]
            encoder.save(col_path)

    def _load_cb_encoder(self, path, name="encoder.pkl") -> CBDataEncoder:
        if os.path.exists(path):
            try:
                with open(os.path.join(path, name), "rb") as fp:
                    return pickle.load(fp)
            except FileNotFoundError as e:
                logging.error("No encoder found: %s", e)
                raise e

    def _load_encoder(self):
        pass
