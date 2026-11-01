# FastAPI app entrypoint - Day 13
"""
ragregress/main.py

FastApi app.   `uvicorn ragregress.main:app` will run it


What the tests need (tests/integration/test_api.py, fixture `api`):
 
    create_app(settings=None, executor=execute_run, configure=True) -> FastAPI
 
  - settings None  -> ApiSettings()             (the defaults; the tests always pass their own temp settings)
  - one JobManager(executor) per app            (steps 1-5 give it to the runs router; step 1 does not need it yet)
  - start-up (a FastAPI "lifespan"), in this order:
        if configure:  load_dotenv(find_dotenv(usecwd=True))  and  configure_logging()
        always:        init_db(settings.db_path)              # creates the SQLite file and tables
    `configure=False` is how the tests keep your real .env and logging untouched.
  - include the health router now; the experiments router joins in step 2.
 
At the bottom, once everything works (step 6):   app = create_app()
 
Things to read in the FastAPI docs: "Lifespan Events", "Bigger Applications - Multiple Files" (APIRouter).
Imports you will want:
    from contextlib import asynccontextmanager
    from dotenv import find_dotenv, load_dotenv
    from fastapi import FastAPI
    from ragregress.core.logging import configure_logging
    from ragregress.storage.database import init_db

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