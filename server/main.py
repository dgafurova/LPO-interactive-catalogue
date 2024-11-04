from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from internal.app.config import api_v1
from internal.controller.router import router
from starlette.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from internal.model.error import NotFoundError, NotValidError
from fastapi.responses import JSONResponse


app = FastAPI()

app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

# front_app = FastAPI()

# front_app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")

# app.mount("/api/v1", app)


@app.get("/api/v1/healthcheck", status_code=200)
def healthcheck():
    return


@app.get("/docs")
def read_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json")


@app.exception_handler(NotFoundError)
async def not_found_exception_handler(request, exc: NotFoundError):
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc)},
    )


@app.exception_handler(NotValidError)
async def not_found_exception_handler(request, exc: NotValidError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


# app.include_router(router, prefix=api_v1)
app.include_router(router, prefix=api_v1)

app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

app.mount("/", StaticFiles(directory="dist", html=True), name="frontend")
