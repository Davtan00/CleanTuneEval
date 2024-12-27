import uvicorn
from api.main import app
from api.endpoints import router

# Register routes
app.include_router(router)

def run_server(host: str = "localhost", port: int = 8000):
    """
    Run the FastAPI server
    """
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        reload=True  
    )
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_server() 