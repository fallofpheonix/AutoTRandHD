from autotrandhd.api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("autotrandhd.api.app:app", host="0.0.0.0", port=8000)
