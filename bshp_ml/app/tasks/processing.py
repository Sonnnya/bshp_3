import logging

from schemas.models import ModelStatuses

from .loader import DataLoader

from ml.models import ModelManager  # noqa: F401  # isort: skip
from .manager import TaskManager

logger = logging.getLogger(__name__)


async def process_uploading_task(
    task_manager: TaskManager, data_loader: DataLoader, task_id: str
):
    """Background task for uploading data from file."""

    logger.info(f"[{task_id}] process_uploading_task started")

    try:
        task = await task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Execute loading
        result = await data_loader.upload_data_from_file(
            task_manager,
            task,
        )

        logger.info(f"[{task_id}] uploading task completed")

        # Update status
        await task_manager.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in data loading task: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))


async def process_fitting_model(
    task_manager: TaskManager, model_manager: ModelManager, task_id: str
):
    """Background task for fitting model."""

    # TODO: this

    logger.info(f"[{task_id}] process_fitting_task started")

    try:
        task = await task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Execute loading
        model = await model_manager.get_model(task.model_type, task.base_name)
        model_manager.add_model(model)
        if model.status == ModelStatuses.FITTING:
            raise ValueError("Current model is already fitting")

        parameters = task.parameters
        if not parameters:
            parameters = {}

        if "data_filter" not in parameters:
            parameters["data_filter"] = {}

        data_filter = (
            {"base_name": task.base_name} if task.base_name != "all_bases" else None
        )  # TODO: .get()?
        if data_filter:
            parameters["data_filter"].update(data_filter)

        await model.fit(parameters)

        logger.info("Start writing model to db")
        try:
            await model_manager.write_model(model)
        except Exception as e:
            model.status = ModelStatuses.ERROR
            raise e
        logger.info("Writing model to db. Done")

        # Update status
        logger.info(f"[{task_id}] fitting task completed")
        await task_manager.update_task(task_id, status="READY", progress=100)
        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in model fitting task: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))

    finally:
        await model_manager.release_training_slot()


async def process_fitting_model_v2(
    task_manager: TaskManager, model_manager: ModelManager, task_id: str, Xy_api: dict
):
    """Background task for fitting model."""

    # TODO: this

    logger.info(f"[{task_id}] process_fitting_task started")

    try:
        task = await task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Execute loading
        model = await model_manager.get_model(task.model_type, task.base_name)
        model_manager.add_model(model)
        if model.status == ModelStatuses.FITTING:
            raise ValueError("Current model is already fitting")

        parameters = task.parameters
        if not parameters:
            parameters = {}

        if "data_filter" not in parameters:
            parameters["data_filter"] = {}

        data_filter = (
            {"base_name": task.base_name} if task.base_name != "all_bases" else None
        )  # TODO: .get()?
        if data_filter:
            parameters["data_filter"].update(data_filter)

        try:
            await model.fit(Xy_api=Xy_api, parameters=parameters)
            logger.info("Start writing model to db")
            await model_manager.write_model(model)
        except Exception as e:
            model.status = ModelStatuses.ERROR
            raise e
        logger.info("Writing model to db. Done")

        # Update status
        logger.info(f"[{task_id}] fitting task completed")
        await task_manager.update_task(task_id, status="READY", progress=100)
        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in model fitting task: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))

    finally:
        await model_manager.release_training_slot()
