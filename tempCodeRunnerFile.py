from fastapi import FastAPI, Request, UploadFile, File, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
import pandas as pd
import os
from langchain_cerebras import ChatCerebras
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv
import tempfile
import uuid
from datetime import datetime
from starlette.middleware.sessions import SessionMiddleware
import shutil

# Load environment variables
load_dotenv()

# Create FastAPI instance
app = FastAPI()

# Add session middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

# Create upload directories
UPLOAD_FOLDER = "uploads"
WORKFLOW_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "workflows")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORKFLOW_UPLOAD_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize global variables
df = None
agent_executor = None

# Initialize the model
model = ChatCerebras(
    model="llama3.3-70b",
    temperature=0
)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Drishti.AI - Chat", "active_page": "chat"}
    )

@app.get("/workflows")
async def workflows_page(request: Request):
    return templates.TemplateResponse(
        "workflows.html",
        {"request": request, "title": "Drishti.AI - Workflows", "active_page": "workflows"}
    )

@app.get("/workflows/data-cleaning")
async def data_cleaning_page(request: Request):
    # Initialize session if needed
    if "workflow_session_id" not in request.session:
        request.session["workflow_session_id"] = str(uuid.uuid4())
        request.session["workflow_history"] = []
    
    return templates.TemplateResponse(
        "data_cleaning.html",
        {
            "request": request,
            "title": "Drishti.AI - Data Cleaning",
            "active_page": "workflows",
            "history": request.session.get("workflow_history", []),
            "has_file": os.path.exists(os.path.join(
                WORKFLOW_UPLOAD_FOLDER,
                f"{request.session['workflow_session_id']}_workflow_data.csv"
            )) if "workflow_session_id" in request.session else False
        }
    )

@app.post("/workflows/data-cleaning/upload")
async def upload_workflow_file(request: Request, file: UploadFile = File(...)):
    # Initialize session if needed
    if "workflow_session_id" not in request.session:
        request.session["workflow_session_id"] = str(uuid.uuid4())
        request.session["workflow_history"] = []
    
    session_id = request.session["workflow_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_workflow_data.csv")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify the file is a valid CSV
        df = pd.read_csv(file_path)
        
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "preview": df.head().to_html(classes='table table-striped'),
            "rows": len(df),
            "columns": len(df.columns)
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/data-cleaning/action")
