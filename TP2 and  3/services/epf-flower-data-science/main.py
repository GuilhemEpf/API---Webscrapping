import uvicorn

from src.app import get_application
from fastapi.responses import RedirectResponse

app = get_application()

# Redirection de l'endpoint racine vers la documentation Swagger
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/hello", tags=["General"])
def say_hello():
    return {"message": "Hello, FastAPI!"}


if __name__ == "__main__":
    uvicorn.run("main:app", debug=True, reload=True, port=8000)








