import asyncio
import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import pandas as pd
from cachetools import TTLCache
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.exceptions import HTTPException
from fastapi.responses import HTMLResponse  # FileResponse
from ml import ModelManager, get_model_manager
from ml import init_manager as init_model_manager
from schemas.models import (
    DataRow,
    ModelInfo,
    ModelTypes,
)
from schemas.tasks import (
    StatusResponse,
    TaskResponse,
)
from settings import AUTH_SERVICE_URL, DB_URL, TEST_MODE, VERSION
from tasks import task_manager
from tasks.__init__ import data_loader
from tasks.processing import process_fitting_model, process_uploading_task

from db import db_processor

pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)

logger = logging.getLogger(__name__)

auth_cache = TTLCache(maxsize=1000, ttl=300)


async def check_token(token: str = Header()) -> bool:
    """Проверка токена авторизации с кэшем и защитой от сбоев."""

    if TEST_MODE:
        return True

    if token in auth_cache:
        return True

    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{AUTH_SERVICE_URL}/check_token",  # type: ignore
                json={"token": token},
            ) as response:
                if response.status == 200:
                    auth_cache[token] = True
                    return True
                elif response.status == 404:
                    raise HTTPException(status_code=401, detail="Invalid token")
                elif response.status == 403:
                    raise HTTPException(status_code=401, detail="Token expired")
                else:
                    logger.error(
                        f"Auth service returned unexpected status {response.status}"
                    )
                    raise HTTPException(
                        status_code=500, detail="Authentication service error"
                    )

    except asyncio.TimeoutError:
        logger.error("Auth request timed out")
        raise HTTPException(status_code=503, detail="Authorization timeout")

    except aiohttp.ClientError as e:
        logger.warning(f"Auth service connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    except Exception:
        logger.exception("Unexpected error in check_token")
        raise HTTPException(status_code=500, detail="Internal auth error")


async def get_token_from_header(token: Optional[str] = Header(None, alias="token")):
    return token


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting DB connection")
        app.db = db_processor
        logger.info(DB_URL)
        await app.db.connect(url=DB_URL)
        await app.db.delete_temp_collections()
        logger.info("DB connection done")

    except Exception as e:
        logger.error(f"DB startup error: {e}")
        raise

    try:
        logger.info("Starting models initialize")
        init_model_manager()

        await get_model_manager().read_models()
        logger.info("Models initializing done")
    except Exception as e:
        logger.error(f"Models initializing error: {e}")
        raise

    yield

    try:
        app.db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise

    logger.info("Shutting down prediction service...")


# if os.getenv("DEBUG") == "true":
#     import debugpy

#     debugpy.listen(("0.0.0.0", 5678))
#     logger.info("Debugpy is listening on 5678")
#     debugpy.wait_for_client()

app = FastAPI(
    title="BSHP App",
    description="Application for AI cash flow parameters predictions!",
    version=VERSION,
    lifespan=lifespan,
)


@app.get("/")
async def main_page():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return HTMLResponse("<h2>BSHP module</h2> <br> <h3>Connection established</h3>")


@app.get("/health")
def health() -> str:
    return "service is working"


@app.get("/version")
def version() -> str:
    return VERSION


@app.post("/save_data")
async def save_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    base_name: str = Query(),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
    replace: bool = Query(default=False),
) -> TaskResponse:
    logger.info(
        f"Starting uploading from file: {file.filename},\n will be available with base_name = {base_name}"
    )

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:
        # Save file
        content = await file.read()
        file_path = await task_manager.save_upload_file(task_id, file.filename, content)  # type: ignore

        # Update task
        await task_manager.update_task(
            task_id,
            type="UPLOAD",
            status="UPLOADING_FILE",
            upload_progress=100,
            file_path=str(file_path),
            base_name=base_name,
            replace=replace,
        )

        # Start background task
        background_tasks.add_task(
            process_uploading_task, task_manager, data_loader, task_id
        )

        return TaskResponse(task_id=task_id, message="Task processing started")

    except Exception as e:
        logger.error(f"Error in uploading task {task_id}: {e}")

        if "File name too long" in str(e):
            await task_manager.update_task(
                task_id, status="ERROR", error="Имя файла слишком длинное"
            )
            await task_manager.update_task(
                task_id, status="ERROR", error="Имя файла слишком длинное"
            )
            raise HTTPException(
                status_code=500,
                detail="Имя файла слишком длинное. Сократите имя файла и попробуйте загрузить ещё раз",
            )
        else:
            await task_manager.update_task(task_id, status="ERROR", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/delete_data")
async def delete_data(
    base_name: str = Query(default=""),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
) -> str:
    db_filter = None

    if base_name:
        db_filter = {"base_name": base_name}
        db_filter = {"base_name": base_name}
    await data_loader.delete_data(db_filter)
    return "Data has been deleted"


# @app.post("/fit")
@app.post("/v1/fit")
async def fit(
    background_tasks: BackgroundTasks,
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.rf),
    parameters: dict = Body(),
    model_manager=Depends(get_model_manager),
) -> TaskResponse:
    logger.info("Start fitting model")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

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


@app.get("/delete_model")
async def delete_model(
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.catboost_txt),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
    model_manager: ModelManager = Depends(get_model_manager),
) -> str:
    try:
        model_type = ModelTypes.catboost_txt
        await model_manager.delete_model(model_type=model_type, base_name=base_name)
        return "Model has been deleted"
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_model_info")
async def get_model_info(
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.rf),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
    model_manager=Depends(get_model_manager),
) -> ModelInfo:
    try:
        model_type = ModelTypes.catboost_txt

        result = await model_manager.get_info(
            model_type=model_type.value, base_name=base_name
        )
        return ModelInfo.model_validate(result)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/predict")
@app.post("/v1/predict")
async def predict(
    X: list[DataRow],
    base_name: str = Query(default=""),
    model_type: ModelTypes = Query(default=ModelTypes.rf),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
    model_manager=Depends(get_model_manager),
) -> list[DataRow]:
    try:
        if not base_name:
            base_name = "all_bases"
        X_list = []
        for row in X:
            X_list.append(row.model_dump())
        model = await model_manager.get_model(model_type, base_name, log=False)
        result = []
        X_y_list = await model.predict(X_list)
        for row in X_y_list:
            result.append(DataRow.model_validate(row))
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result


@app.get("/get_task_status")
async def get_task_status(
    task_id: str = Query(),
    token: str = Depends(get_token_from_header),
    authenticated: bool = Depends(check_token),
) -> StatusResponse:
    status = await task_manager.get_task_status(task_id)

    return status


from api import cb_router, embed_router

app.include_router(embed_router)
app.include_router(cb_router)
