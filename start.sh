#!/bin/bash
# Start FastAPI using Uvicorn on the port Render provides
bash -lc "uvicorn server:app --host=0.0.0.0 --port=$PORT"
