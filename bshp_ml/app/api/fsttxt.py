import json
import logging
import traceback
import uuid

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from ml.models import ModelManager, get_model_manager
from schemas.models import ExtDataRow, ModelTypes
from schemas.tasks import TaskResponse
from tasks.__init__ import task_manager
from tasks.processing import process_fitting_model

router = APIRouter(
    prefix="/embeddings", tags=["Embedding модель (по умолчанию FastText)"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)


@router.post("/fit")
async def fit_embeddings(
    background_tasks: BackgroundTasks,
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    parameters: dict = Body(),
    model_manager: ModelManager = Depends(get_model_manager),
) -> TaskResponse:
    model_type = ModelTypes.fstxt
    task_id = str(uuid.uuid4())
    # should be through depends
    task = await task_manager.create_task(task_id)
    logger.info(
        f"Start fitting model {model_type} on {base_name}, TASK ID {task.task_id}"
    )

    try:
        if not base_name:
            base_name = "all_bases"
        # Update task
        await task_manager.update_task(
            task_id,
            type="FIT",
            status="PREPARE_FITTING",
            upload_progress=100,
            model_type=model_type.value,
            base_name=base_name,
            parameters=parameters,
        )

        # Reject if too many trainings are already running (released in bg task)
        if not await model_manager.acquire_training_slot():
            raise HTTPException(
                status_code=429,
                detail="Max concurrent training jobs reached. Try again later.",
                headers={"Retry-After": "60"},
            )

        # Start background task
        background_tasks.add_task(
            process_fitting_model, task_manager, model_manager, task_id
        )

        return TaskResponse(task_id=task_id, message="Task processing started")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_embeddings(
    X: list[ExtDataRow],
    base_name: str = Query(default="all_bases"),
    # token: str = Depends(get_token_from_header),
    # authenticated: bool = Depends(check_token),
    model_manager: ModelManager = Depends(get_model_manager),
    # ) -> list[ExtDataRow]:
) -> list[dict]:
    # пока что без других эмбеддинг моделей
    model_type = ModelTypes.fstxt
    # TODO: depends get_model
    try:
        if not base_name:
            base_name = "all_bases"
        # X_list = []
        # for row in X:
        #     X_list.append(row.model_dump())

        model = model_manager._sync_get_model(model_type, base_name)
        result = []
        X_y_list = json.loads(await model.predict(jsonable_encoder(X)))
        result = X_y_list
        # for row in X_y_list:
        #     # TODO: можно быстрее?
        #     result.append(EmbedPredictionsRow.model_validate(row))
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result
