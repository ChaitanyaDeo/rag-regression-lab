"""
ragregress/main.py

The FastAPI app. Serve it with:  uvicorn ragregress.main:app

create_app(settings=None, executor=execute_run, configure=True) -> FastAPI
  settings   ApiSettings; None means the defaults.
  executor   runs one experiment in the background; tests pass a fake.
  configure  True: load .env and set up logging at start-up. Tests pass False.

Start-up (lifespan): load_dotenv + configure_logging (only if configure), then init_db (always).
Routers: health (GET /health) and experiments (runs, compare, reports).
"""

from __future__ import annotations
 
from fastapi import FastAPI
from dotenv import find_dotenv, load_dotenv

from contextlib import asynccontextmanager

from ragregress.api import health
from ragregress.api.jobs import JobManager, execute_run
from ragregress.api.settings import ApiSettings

from ragregress.storage.database import init_db
from ragregress.api import experiments
from ragregress.core.logging import configure_logging



def create_app(settings=None, executor = execute_run, configure= True) -> FastAPI:

    settings = settings or ApiSettings()

    jobs = JobManager(executor)

    @asynccontextmanager
    async def lifespan(app):
        if configure:
            load_dotenv(find_dotenv(usecwd=True))
            configure_logging()
        init_db(settings.db_path)

        yield


    app = FastAPI(title="RAG Regression Lab", lifespan=lifespan)

    app.include_router(health.router)

    app.include_router(experiments.build_router(settings, jobs))

    return app

app = create_app()