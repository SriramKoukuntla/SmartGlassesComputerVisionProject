"""Simple script to start the FastAPI server."""
import uvicorn

if __name__ == "__main__":
    print("Starting Smart Glasses API Server...")
    print("API will be available at http://localhost:8000")
    print("Frontend should connect to http://localhost:8000")
    print("\nPress Ctrl+C to stop the server\n")
    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

