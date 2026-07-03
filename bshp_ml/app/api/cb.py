import json
import logging
import traceback
import uuid
from typing import Optional

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    Header,
    HTTPException,
    Query,
)
from fastapi.encoders import jsonable_encoder
from ml.models import ModelManager, get_model_manager
from schemas.models import ExtDataRow, ModelTypes
from schemas.tasks import TaskResponse
from settings import USE_DETAILED_LOG, settings
from tasks.__init__ import Reader, task_manager
from tasks.processing import (
    process_fitting_model,
    process_fitting_model_v2,
)

logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/v2", tags=["Новая cb"])
router = APIRouter()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)


async def get_token_from_header(token: Optional[str] = Header(None, alias="token")):
    return token


@router.post("/fit")
async def fit(
    background_tasks: BackgroundTasks,
    token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    parameters: dict = Body(),
    model_manager: ModelManager = Depends(get_model_manager),
    fit_embeddings: bool = False,
    model_type: ModelTypes = Query(
        default=ModelTypes.catboost_txt
    ),  # ни на что не влияет, всегда эта модель
):
    parameters["calculate_metrics"] = (
        False  # forced, breaks otherwise. metrics are calculated and stored in csv for now
    )
    parameters["use_cross_validation"] = False  # forced
    # TODO: fix in frontend, no need to send these parameters at all, they are not used in current implementation of metrics calculation
    # what was the logic behind this anyway?

    # 1. Fit embeddings first
    # TODO: 404 модель не найдена
    if fit_embeddings:
        task_id = str(uuid.uuid4())
        task = await task_manager.create_task(task_id)
        logger.info(
            f"Start fitting model {ModelTypes.extfstxt} on {'all_bases'}, TASK ID {task.task_id}"
        )

        try:
            await task_manager.update_task(
                task_id,
                type="FIT",
                status="PREPARE_FITTING",
                upload_progress=100,
                model_type=ModelTypes.extfstxt.value,
                base_name="all_bases",
                parameters=parameters,
            )
            # Reject if too many trainings are already running (released in bg task)
            if not await model_manager.acquire_training_slot():
                raise HTTPException(
                    status_code=429,
                    detail=f"Max models limit reached ({settings.MAX_MODELS} models training: {', '.join(model_manager.get_training_models_names())}). Try again later.",
                    headers={"Retry-After": "60"},
                )
            background_tasks.add_task(
                process_fitting_model, task_manager, model_manager, task_id
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error while fitting embeddings model: {e}")

            await task_manager.update_task(task_id, status="ERROR", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # 2. Predict values with embeddings and pass it to catboost model
    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    # Reject before reading data from the DB if too many trainings are already
    # running, to keep RAM free. Released on any failure below, or by the bg task.
    if not await model_manager.acquire_training_slot():
        raise HTTPException(
            status_code=429,
            detail=f"Max models limit reached ({settings.MAX_MODELS} models training: {', '.join(model_manager.get_training_models_names())}). Try again later.",
            headers={"Retry-After": "60"},
        )

    try:
        X_y = await _read_dataset({"data_filter": {"base_name": base_name}})
        if USE_DETAILED_LOG:
            logging.info("Loading columns: %s, %s", X_y.columns, X_y.shape)
        if X_y.empty:
            raise ValueError("No data. Save the data from 1C")
    except Exception as e:
        await model_manager.release_training_slot()
        print(traceback.format_exc())
        logger.error(
            f"Collection not found. Please, insure base {base_name} is loaded: {e}"
        )
        raise HTTPException(
            status_code=404, detail=f"No data. Save the data from 1C. Detail: {str(e)}"
        )

    try:
        fsttext = await model_manager.get_model(ModelTypes.extfstxt, "all_bases")
        # модель фасттекст для всех баз одна
        Xy_embed = await fsttext.predict(X_y, set_classes=True)
        Xy_json = json.loads(Xy_embed)
    except Exception as e:
        await model_manager.release_training_slot()
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    # 3. Use fsttext model predictions for training catboost model,
    # categories, float_columns = fsttxt.columns

    model_type = ModelTypes.catboost_txt
    logger.info(f"Start fitting model {model_type.value}")

    try:
        await task_manager.update_task(
            task_id,
            type="FIT",
            status="PREPARE_FITTING",
            upload_progress=100,
            model_type=model_type.value,
            base_name=base_name,
            parameters=parameters,
        )

        # Start background task (slot acquired above is released in the bg task)
        background_tasks.add_task(
            process_fitting_model_v2, task_manager, model_manager, task_id, Xy_json
        )

        return TaskResponse(task_id=task_id, message="Task processing started")

    except Exception as e:
        # Scheduling failed after the slot was acquired — release it.
        await model_manager.release_training_slot()
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        model_manager.unload_model(model_type, base_name)


@router.post("/predict")
async def predict(
    X: list[ExtDataRow],
    base_name: str = Query(default=""),
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    model_manager: ModelManager = Depends(get_model_manager),
    model_type: ModelTypes = Query(
        default=ModelTypes.catboost_txt
    ),  # ни на что не влияет, всегда эта модель
):
    # 1. Predict values with embeddings and pass it to catboost model
    try:
        dataset = await _read_dataset({"data_filter": {"base_name": base_name}})
        if USE_DETAILED_LOG:
            logging.info("Loading columns: %s, %s", dataset.columns, dataset.shape)
        if dataset.empty:
            raise ValueError("No data. Save the data from 1C")
    except Exception as e:
        print(traceback.format_exc())
        logger.error(
            f"Collection not found. Please, insure base {base_name} is loaded: {e}"
        )
        raise HTTPException(status_code=404, detail=str(e))
    try:
        fsttext = await model_manager.get_model(ModelTypes.extfstxt, "all_bases")
        Xy_embed = await fsttext.predict(
            jsonable_encoder(X), set_classes=True, set_from=dataset
        )
        Xy_json = json.loads(Xy_embed)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    logger.info("Shape of json for predictions: %s", str(pd.DataFrame(Xy_json).shape))
    model_type = ModelTypes.catboost_txt
    try:
        model = await model_manager.get_model(model_type, base_name)
        result = []
        X_y_list = await model.predict(Xy_json)
        result = X_y_list
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        model_manager.unload_model(model_type, base_name)

    return result


async def _read_dataset(parameters: dict) -> pd.DataFrame:
    data_filter = parameters["data_filter"]
    if USE_DETAILED_LOG:
        logger.info("Reading data from db: %s", data_filter)
    reader = Reader()
    X_y = await reader.read(data_filter)

    return X_y
