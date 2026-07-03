import asyncio
import gc
import json
import logging
import os
import pickle
import random
import shutil
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import UTC, datetime
from typing import Optional

import numpy as np
import pandas as pd
from ml.data_processing import (
    Checker,
    DataEncoder,
    FeatureAdder,
    NanProcessor,
    Shuffler,
)
from schemas.models import ModelInfo, ModelStatuses, ModelTypes
from settings import MODEL_FOLDER, USE_DETAILED_LOG, settings
from sklearn.pipeline import Pipeline

from db import db_processor

logging.getLogger("bshp_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class Model(ABC):
    model_type = None

    def __init__(self, base_name):
        self.base_name = base_name

        self.uid = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                "{}_{}".format(
                    self.model_type.value if self.model_type else "", self.base_name
                ),
            )
        )
        # TODO: в pydantic
        self.x_columns = [
            "is_reverse",
            "document_month",
            "document_year",
            "moving_type",
            "company_inn",
            "company_kpp",
            "base_document_kind",
            "base_document_operation_type",
            "contractor_name",
            "contractor_inn",
            "contractor_kpp",
            "contractor_kind",
            "article_name",
            "is_main_asset",
            "analytic",
            "analytic2",
            "analytic3",
            "article_parent",
            "article_group",
            "article_kind",
            "store",
            "department",
            "company_account_number",
            "contractor_account_number",
            "qty",
            "price",
            "sum",
        ]
        self.y_columns = ["cash_flow_item_code", "cash_flow_details_code", "year"]

        self.additional_columns = [
            "number",
            "date",
        ]

        self.str_columns = [
            "moving_type",
            "company_inn",
            "company_kpp",
            "base_document_kind",
            "base_document_operation_type",
            "contractor_name",
            "contractor_inn",
            "contractor_kpp",
            "contractor_kind",
            "article_name",
            "analytic",
            "analytic2",
            "analytic3",
            "article_parent",
            "article_group",
            "article_kind",
            "store",
            "department",
            "company_account_number",
            "contractor_account_number",
            "cash_flow_item_code",
            "year",
            "cash_flow_details_code",
        ]
        self.bool_columns = ["is_reverse", "is_main_asset"]
        self.float_columns = [
            "qty",
            "price",
            "sum",
        ]

        self.date_columns = [
            "date",
            "base_document_date",
            "article_document_date",
            "uploading_date",
        ]

        self.parameters = {
            "x_columns": self.x_columns,
            "y_columns": self.y_columns,
            "str_columns": self.str_columns,
            "float_columns": self.float_columns,
            "bool_columns": self.bool_columns,
            "additional_columns": self.additional_columns,
        }

        self.status = ModelStatuses.CREATED
        self.error_text = ""

        self.fitting_start_date: Optional[datetime] = None
        self.fitting_end_date: Optional[datetime] = None

        self.metrics = {}

        self.columns_to_encode = [
            "is_reverse",
            "moving_type",
            "company_inn",
            "company_kpp",
            "base_document_kind",
            "base_document_operation_type",
            "contractor_name",
            "contractor_inn",
            "contractor_kpp",
            "contractor_kind",
            "article_name",
            "is_main_asset",
            "analytic",
            "analytic2",
            "analytic3",
            "article_parent",
            "article_group",
            "article_kind",
            "store",
            "department",
            "company_account_number",
            "contractor_account_number",
            "cash_flow_item_code",
            "year",
            "cash_flow_details_code",
        ]
        self.parameters["columns_to_encode"] = self.columns_to_encode

        self.field_models = {}
        self.test_field_models = {}
        self.data_encoder = None

        self.strict_acc = {}
        self.test_strict_acc = {}
        self.need_to_encode = True
        self.classes = {}

        self.metrics_dataset_name = ""
        self.test_metrics_dataset_name = ""
        self.is_loaded = False

    async def fit(self, parameters):
        logger.info("Fitting")
        try:
            need_to_initialize = (
                self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR]
                or parameters.get("refit") == 0
            )
            calculate_metrics = parameters.get("calculate_metrics")
            use_cross_validation = parameters.get("use_cross_validation")

            await self._before_fit(
                parameters, need_to_initialize, calculate_metrics, use_cross_validation
            )
            X_y = await self._read_dataset(parameters)

            train_test_indexes = None
            self.metrics_dataset_name = ""
            self.test_metrics_dataset_name = ""
            if use_cross_validation:
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

    @abstractmethod
    async def _fit(self, dataset, parameters, is_first=True): ...

    @abstractmethod
    async def predict(self, X, for_metrics=False): ...

    def _get_train_test_indexes(self, X_y):
        indexes_len = X_y.shape[0]
        indexes = list(np.arange(indexes_len))
        random.shuffle(indexes)

        test_size = 0.2

        train_indexes = indexes[: int(indexes_len * (1 - test_size))]
        test_indexes = indexes[int(indexes_len * (1 - test_size)) :]

        return train_indexes, test_indexes

    async def _get_train_test_datasets(
        self, X_y, train_indexes, test_indexes, calculate_metrics, use_cross_validation
    ):
        train_dataset = X_y.iloc[train_indexes]
        test_dataset = X_y.iloc[test_indexes]

        to_add = []
        to_delete = []
        for y_col in self.y_columns:
            train_values = set(train_dataset[y_col].unique())
            test_values = set(test_dataset[y_col].unique())

            for val in train_values:
                if val not in test_values:
                    to_add.append(val)

            for val in test_values:
                if val not in train_values:
                    to_delete.append(val)

            if to_delete:
                test_dataset = test_dataset.loc[~test_dataset[y_col].isin(to_delete)]

            if to_add:
                add_dataset = train_dataset.loc[train_dataset[y_col].isin(to_add)]
                test_dataset = pd.concat([test_dataset, add_dataset])

        return train_dataset, test_dataset

    async def _get_metric(self, dataset):
        logger.info("Get metrics")
        metrics = -1

        c_x_columns = self.x_columns
        dataset["check"] = 0

        to_rename = {el: "{}_val".format(el) for el in self.y_columns}
        dataset = dataset.rename(to_rename, axis=1)

        dataset_pred = await self.predict(dataset, for_metrics=True)

        for y_col in self.y_columns:
            check_column = "{}_check".format(y_col)
            pred_column = "{}_pred".format(y_col)
            val_column = "{}_val".format(y_col)

            dataset[pred_column] = dataset_pred[y_col]
            logger.info('Calculating metrics , field "{}"'.format(y_col))

            dataset[check_column] = dataset[val_column] != dataset[pred_column]

            dataset["check"] = dataset["check"] + dataset[check_column]
            logger.info("Done")

            dataset["check"] = dataset["check"] == 0
            dataset["check"] = dataset["check"].astype(int)

            dataset["row"] = 1

            dataset_grouped = (
                dataset[["number", "date", "row", "check"]]
                .groupby(by=["number", "date"])
                .sum()
            )
            dataset_grouped["check_all"] = (
                dataset_grouped["row"] == dataset_grouped["check"]
            )
            dataset_grouped["check_all"] = dataset_grouped["check_all"].astype(int)

            all = dataset_grouped["check_all"].shape[0]
            right = dataset_grouped[dataset_grouped["check_all"] == 1].shape[0]
            metrics = right / all

        return metrics

    async def _before_fit(
        self, parameters, need_to_initialize, calculate_metrics, use_cross_validation
    ):
        self.status = ModelStatuses.FITTING
        self.fitting_start_date = datetime.now(UTC)

        if need_to_initialize:
            self.__init__(self.base_name)
            self.need_to_encode = parameters.get("need_to_encode", True)
            self._delete_all_models()

    async def _after_fit(self, parameters, need_to_initialize, use_cross_validation):
        await self.save()
        if self.metrics_dataset_name:
            await self._delete_dataset_from_temp(self.metrics_dataset_name)
            self.metrics_dataset_name = ""
        if self.test_metrics_dataset_name:
            await self._delete_dataset_from_temp(self.test_metrics_dataset_name)
            self.test_metrics_dataset_name = ""
        self.status = ModelStatuses.READY
        self.fitting_end_date = datetime.now(UTC)
        logger.info("Fitting. Done")

    async def _read_dataset(self, parameters, limited=False) -> pd.DataFrame:
        from tasks.__init__ import Reader

        data_filter = parameters["data_filter"]
        if USE_DETAILED_LOG:
            logger.info("Reading data from db")
        reader = Reader()
        if limited:
            X_y = await reader.read_limited(data_filter)
        else:
            X_y = await reader.read(data_filter)

        return X_y

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
        if self.need_to_encode:
            if need_to_initialize:
                self.data_encoder = DataEncoder(self.parameters)
            self.data_encoder.form_encode_dict = need_to_initialize
            pipeline_list.append(("data_encoder", self.data_encoder))

        pipeline_list.append(("shuffler", Shuffler(self.parameters)))

        pipeline = Pipeline(pipeline_list)
        dataset = pipeline.fit_transform(dataset)

        datasets = {}

        if train_test_indexes:
            datasets["train"] = dataset.iloc[train_test_indexes[0]]
            datasets["test"] = dataset.iloc[train_test_indexes[1]]
        else:
            datasets["train"] = dataset

        return datasets

    async def _on_fitting_error(self, ex):
        self.status = ModelStatuses.ERROR
        self.error_text = str(ex)
        if self.metrics_dataset_name:
            await self._delete_dataset_from_temp(self.metrics_dataset_name)
            self.metrics_dataset_name = ""
        if self.test_metrics_dataset_name:
            await self._delete_dataset_from_temp(self.test_metrics_dataset_name)
            self.test_metrics_dataset_name = ""

        await self.save(without_models=True)

        raise ex

    async def _calculate_metrics(
        self, parameters, need_to_initialize=False, use_cross_validation=False
    ):
        if USE_DETAILED_LOG:
            logger.info("Start calculating metrics")
        dataset = await self._load_dataset_from_temp(self.metrics_dataset_name)
        self.metrics["train"] = await self._get_metric(dataset)
        if use_cross_validation:
            test_dataset = await self._load_dataset_from_temp(
                self.test_metrics_dataset_name
            )
            self.metrics["test"] = await self._get_metric(test_dataset)

        if USE_DETAILED_LOG:
            logger.info("Calculating metrics. Done")

    @abstractmethod
    def _save_column_model(self, column, item=None): ...

    @abstractmethod
    def _load_column_model(self, column, item=None): ...

    def _save_parameters(self):
        path_to_model = os.path.join(MODEL_FOLDER, self.uid)
        if not os.path.isdir(path_to_model):
            os.makedirs(path_to_model)

        parameters = deepcopy(self.parameters)
        parameters["uid"] = self.uid
        parameters["model_type"] = self.model_type.value
        parameters["base_name"] = self.base_name
        parameters["metrics"] = self.metrics
        parameters["need_to_encode"] = self.need_to_encode
        parameters["status"] = self.status.value
        parameters["classes"] = {
            k: [int(el) for el in v] for k, v in self.classes.items()
        }
        with open(os.path.join(path_to_model, "parameters.json"), "w") as fp:
            json.dump(parameters, fp)

    def _load_parameters(self):
        with open(os.path.join(MODEL_FOLDER, self.uid, "parameters.json"), "r") as fp:
            parameters = json.load(fp)

        self.parameters = parameters

        self.base_name = parameters["base_name"]

        self.x_columns = parameters["x_columns"]
        self.y_columns = parameters["y_columns"]

        self.additional_columns = parameters["additional_columns"]

        self.str_columns = parameters["str_columns"]
        self.bool_columns = parameters["bool_columns"]
        self.float_columns = parameters["float_columns"]

        self.metrics = parameters["metrics"]
        self.status = ModelStatuses(parameters["status"])

        self.columns_to_encode = parameters["columns_to_encode"]

        self.need_to_encode = parameters["need_to_encode"]
        self.classes = {
            k: [np.int64(el) for el in v] for k, v in parameters["classes"].items()
        }

    def _save_encoder(self):
        if self.need_to_encode:
            path_to_model = os.path.join(MODEL_FOLDER, self.uid)
            if not os.path.isdir(path_to_model):
                os.makedirs(path_to_model)

            with open(os.path.join(path_to_model, "encoder.pkl"), "wb") as fp:
                pickle.dump(self.data_encoder, fp)

    def _load_encoder(self):
        if self.need_to_encode:
            path_to_model = os.path.join(MODEL_FOLDER, self.uid)
            if os.path.exists(path_to_model):
                with open(os.path.join(path_to_model, "encoder.pkl"), "rb") as fp:
                    self.data_encoder = pickle.load(fp)

    async def load_metadata(self, uid: str):
        """Load parameters only — no model weights. Fast, low-memory startup."""
        self.uid = uid
        self._load_parameters()
        if USE_DETAILED_LOG:
            logger.info("Loaded metadata for %s %s", self.model_type, self.base_name)

    def _delete_all_models(self):
        path_to_dir = os.path.join(MODEL_FOLDER, self.uid)

        if os.path.exists(path_to_dir):
            for y_col in self.y_columns:
                if os.path.isdir(os.path.join(path_to_dir, y_col)):
                    shutil.rmtree(os.path.join(path_to_dir, y_col))
        path_to_encoder = os.path.join(path_to_dir, "encoder.pkl")
        if os.path.exists(path_to_encoder):
            os.remove(path_to_encoder)

    def unload(self):
        """Drop all in-memory model weights. Parameters/metadata are kept."""
        self.field_models = {}
        self.test_field_models = {}
        self.data_encoder = None
        if hasattr(self, "field_encoders"):
            self.field_encoders = {}
        self.is_loaded = False
        gc.collect()
        if USE_DETAILED_LOG:
            logger.info("Unloaded model %s %s", self.model_type, self.base_name)

    async def load(self, uid):
        self.uid = uid
        logger.info("Loaded pretrained %s models", self.model_type)

        self._load_parameters()

        if self.status != ModelStatuses.ERROR:
            for y_col in self.y_columns:
                if y_col == "cash_flow_details_code":
                    for item in self.classes[y_col]:
                        self._load_column_model(y_col, item)
                else:
                    self._load_column_model(y_col)

            self._load_encoder()
        self.is_loaded = True

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
                        self._save_column_model(y_col, item)
                else:
                    self._save_column_model(y_col)
            self._save_encoder()
        self._save_parameters()

    async def _save_dataset_to_temp(self, dataset):
        collection_name = "temp_{}".format(uuid.uuid4())

        def date_to_str(value):
            try:
                return datetime.strftime(value, r"%d.%m.%Y %H:%M:%S")
            except:
                return "01.01.1970 00:00:00"

        for col in self.date_columns:
            dataset[col] = dataset[col].apply(date_to_str)

        await db_processor.insert_many(
            collection_name, dataset.to_dict(orient="records")
        )
        return collection_name

    async def _load_dataset_from_temp(self, collection_name) -> pd.DataFrame:
        dataset = await db_processor.find(collection_name)
        dataset = pd.DataFrame(dataset)

        def str_to_date(value):
            try:
                return datetime.strptime(value, r"%d.%m.%Y %H:%M:%S")
            except:  # noqa: E722
                return datetime(1970, 1, 1, 0, 0, 0)

        for col in self.date_columns:
            dataset[col] = dataset[col].apply(str_to_date)
        return dataset

    async def _delete_dataset_from_temp(self, collection_name):
        await db_processor.delete_many(collection_name)


class ModelManager:
    def __init__(self):
        self.models = []
        self.max_models = (
            settings.MAX_MODELS
        )  # max models for training at the same time
        self.active_training = 0
        self._slot_lock = asyncio.Lock()

    async def acquire_training_slot(self) -> bool:
        """Reserve one training slot. Non-blocking: returns False if already at
        MAX_MODELS. The lock guards check+increment so it stays correct even if
        an await is ever added to the critical section."""
        async with self._slot_lock:
            if self.active_training >= self.max_models:
                logger.warning(
                    "Training slot limit reached (%d/%d active). Rejecting request.",
                    self.active_training,
                    self.max_models,
                )
                return False
            self.active_training += 1
            logger.info(
                "Training slot acquired (%d/%d active).",
                self.active_training,
                self.max_models,
            )
            return True

    def get_training_models_names(self) -> list[str]:
        """Get list of currently training models names for logging purposes."""
        return [
            f"{model['model'].model_type.value}_{model['model'].base_name}"
            for model in self.models
            if model["model"].status == ModelStatuses.FITTING
        ]

    async def release_training_slot(self) -> None:
        async with self._slot_lock:
            if self.active_training > 0:
                self.active_training -= 1
            logger.info(
                "Training slot released (%d/%d active).",
                self.active_training,
                self.max_models,
            )

    async def read_models(self):
        models = []
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        model_dirs = os.listdir(MODEL_FOLDER)

        for model_dir in model_dirs:
            logger.info("Reading models from %s", model_dir)
            if not os.path.isdir(os.path.join(MODEL_FOLDER, model_dir)):
                continue

            path_to_parameters = os.path.join(
                MODEL_FOLDER, model_dir, "parameters.json"
            )
            if not os.path.exists(path_to_parameters):
                if model_dir == "pretrained":
                    logger.info(
                        "Reading pretrained models from %s", os.path.abspath(model_dir)
                    )

                    model = None

                    model = self._get_new_model(ModelTypes.extfstxt, "all_bases")
                    await model.load(
                        uid=str(uuid.uuid4)
                    )  # fsttxt is loaded anyway, from /pretrained, id doest matter

                    if model:
                        models.append(
                            {
                                "model_type": model.model_type,
                                "base_name": model.base_name,
                                "model": model,
                            }
                        )
                        logger.info(
                            "Model %s %s was registred",
                            model.model_type,
                            model.base_name,
                        )
                    else:
                        logger.warning(
                            "Model %s wasn`t registred", os.path.abspath(model_dir)
                        )

                else:
                    continue
            else:
                model = None
                # try:
                with open(path_to_parameters, "r") as fp:
                    parameters = json.load(fp)

                model = self._get_new_model(
                    ModelTypes(parameters["model_type"]), parameters["base_name"]
                )
                await model.load_metadata(parameters["uid"])
                # except Exception as e:
                #     pass

                if model:
                    models.append(
                        {
                            "model_type": model.model_type,
                            "base_name": model.base_name,
                            "model": model,
                        }
                    )

        self.models = models

    def add_model(self, model):
        model_list = [
            el
            for el in self.models
            if el["model_type"] == model.model_type
            and el["base_name"] == model.base_name
        ]

        if not model_list:
            self.models.append(
                {
                    "model_type": model.model_type,
                    "base_name": model.base_name,
                    "model": model,
                }
            )

    async def write_model(self, model):
        await model.save()

    async def get_model(
        self, model_type=ModelTypes.rf, base_name="all_bases", log=True, to_delete=False
    ):
        if log:
            logger.info("Get model with params: %s %s", model_type, base_name)
        model_list = [
            el
            for el in self.models
            if el["model_type"] == model_type and el["base_name"] == base_name
        ]
        if model_list:
            model = model_list[-1]["model"]
        else:
            model = self._get_new_model(model_type, base_name)  # TODO: ?
            return model

        if (
            not model.is_loaded
            and model.status != ModelStatuses.ERROR
            and not to_delete
        ):
            if USE_DETAILED_LOG:
                logger.info(
                    "Model not in memory, loading from disk: %s %s",
                    model_type,
                    base_name,
                )
            await model.load(model.uid)
        return model

    def unload_model(self, model_type=ModelTypes.rf, base_name="all_bases"):
        model_list = [
            el
            for el in self.models
            if el["model_type"] == model_type and el["base_name"] == base_name
        ]
        if model_list:
            model_list[-1]["model"].unload()
            logger.info("Model unloaded: %s %s", model_type, base_name)

    def _sync_get_model(
        self, model_type=ModelTypes.rf, base_name="all_bases", log=True
    ):
        if log:
            logger.info("Get model with params: %s %s", model_type, base_name)
        model_list = [
            el
            for el in self.models
            if el["model_type"] == model_type and el["base_name"] == base_name
        ]
        if model_list:
            model = model_list[-1]["model"]
        else:
            model = self._get_new_model(model_type, base_name)  # TODO: ?
        return model

    def _get_new_model(self, model_type=ModelTypes.rf, base_name="") -> type[Model]:
        sublasses = self._get_all_model_subclasses()
        model_classes = [
            el for el in sublasses if getattr(el, "model_type") == model_type
        ]
        if not model_classes:
            raise ValueError('Model type "{}" not allowed'.format(model_type))
        model_class = model_classes[0]
        return model_class(base_name=base_name)

    def _get_all_model_subclasses(self, model_class=None):
        if not model_class:
            model_class = Model

        subclasses = model_class.__subclasses__()
        result = subclasses.copy()

        for cl in subclasses:
            sub_subclasses = self._get_all_model_subclasses(cl)
            result.extend(sub_subclasses)

        return result

    async def delete_model(self, model_type=ModelTypes.rf, base_name=""):
        c_models = [
            el
            for el in self.models
            if el["model_type"] != model_type and el["base_name"] == base_name
        ]
        if c_models:
            model = c_models[0]
        else:
            model = await self.get_model(model_type, base_name, to_delete=True)

        self.models = [
            el
            for el in self.models
            if el["model_type"] != model_type and el["base_name"] != base_name
        ]

        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        model_dir = os.path.join(MODEL_FOLDER, model.uid)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    async def get_info(self, model_type=ModelTypes.rf, base_name=""):
        model_list = [
            el
            for el in self.models
            if el["model_type"] == model_type and el["base_name"] == base_name
        ]
        if model_list:
            model = model_list[-1]["model"]
        else:
            # Not registered at all — return empty defaults
            return ModelInfo(status=ModelStatuses.ERROR, error_text="Model not found")

        return {
            "status": model.status,
            "error_text": model.error_text,
            "fitting_start_date": model.fitting_start_date,
            "fitting_end_date": model.fitting_end_date,
            "metrics": model.metrics,
        }


model_manager = None


def init_manager() -> ModelManager:
    global model_manager
    model_manager = ModelManager()
    return model_manager


def get_model_manager() -> ModelManager:
    return model_manager
