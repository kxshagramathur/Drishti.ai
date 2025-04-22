from fastapi import FastAPI, Request, UploadFile, File, Form, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse
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
import json
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import numpy as np

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
            "columns": df.columns.tolist(),  # Convert columns to list
            "rows": len(df),
            "columns_count": len(df.columns)
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/data-cleaning/action")
async def perform_cleaning_action(request: Request, action: str = Form(...)):
    if "workflow_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["workflow_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_workflow_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No file uploaded"
        }, status_code=400)
    
    try:
        df = pd.read_csv(file_path)
        initial_rows = int(len(df))  # Convert to regular Python int
        initial_cols = int(len(df.columns))  # Convert to regular Python int
        
        if action == "drop_rows":
            df.dropna(inplace=True)
            affected_rows = int(initial_rows - len(df))  # Convert to regular Python int
            affected_cols = 0
            action_desc = "Dropped rows with missing values"
        elif action == "replace_mean":
            missing_counts = df.isna().sum()
            df.fillna(df.mean(numeric_only=True), inplace=True)
            affected_rows = int(missing_counts.sum())  # Convert to regular Python int
            affected_cols = 0
            action_desc = "Replaced missing values with mean"
        elif action == "remove_duplicates":
            initial_duplicates = int(df.duplicated().sum())  # Convert to regular Python int
            df.drop_duplicates(inplace=True)
            affected_rows = initial_duplicates
            affected_cols = 0
            action_desc = "Removed duplicate rows"
        elif action == "remove_empty_columns":
            initial_empty_cols = int(df.isna().all().sum())  # Convert to regular Python int
            df.dropna(axis=1, how='all', inplace=True)
            affected_rows = 0
            affected_cols = initial_empty_cols
            action_desc = "Removed empty columns"
        else:
            return JSONResponse({
                "status": "error",
                "message": "Invalid action"
            }, status_code=400)
        
        # Save the updated DataFrame
        df.to_csv(file_path, index=False)
        
        # Add to workflow history
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action_desc,
            "initial_rows": initial_rows,
            "initial_cols": initial_cols,
            "affected_rows": affected_rows,
            "affected_cols": affected_cols,
            "final_rows": int(len(df)),  # Convert to regular Python int
            "final_cols": int(len(df.columns))  # Convert to regular Python int
        }
        
        if "workflow_history" not in request.session:
            request.session["workflow_history"] = []
        request.session["workflow_history"].append(history_entry)
        
        return JSONResponse({
            "status": "success",
            "message": action_desc,
            "preview": df.head().to_html(classes='table table-striped'),
            "history": request.session["workflow_history"]
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/workflows/data-cleaning/download")
async def download_cleaned_data(request: Request):
    if "workflow_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["workflow_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_workflow_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No cleaned data available"
        }, status_code=400)
    
    try:
        return FileResponse(
            path=file_path,
            filename="cleaned_data.csv",
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=cleaned_data.csv"
            }
        )
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Error downloading file: {str(e)}"
        }, status_code=500)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df, agent_executor
    
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Read the CSV file
        df = pd.read_csv(temp_file_path)
        
        # Create the agent
        agent_executor = create_pandas_dataframe_agent(
            model,
            df,
            allow_dangerous_code=True,
            verbose=True
        )
        
        # Get a preview of the data
        preview = df.head().to_html(classes='table table-striped')
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return JSONResponse({
            "status": "success",
            "preview": preview,
            "message": "File uploaded successfully"
        })
    except Exception as e:
        # Clean up the temporary file in case of error
        os.unlink(temp_file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/chat")
async def chat(prompt: str = Form(...)):
    global agent_executor
    
    if agent_executor is None:
        return JSONResponse({
            "status": "error",
            "message": "Please upload a CSV file first"
        }, status_code=400)
    
    try:
        response = agent_executor.run(prompt)
        return JSONResponse({
            "status": "success",
            "response": response
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/workflows/heatmap")
async def heatmap_page(request: Request):
    # Initialize session if needed
    if "heatmap_session_id" not in request.session:
        request.session["heatmap_session_id"] = str(uuid.uuid4())
    
    return templates.TemplateResponse(
        "heatmap.html",
        {
            "request": request,
            "title": "Drishti.AI - Heat Map Generation",
            "active_page": "workflows",
            "has_file": os.path.exists(os.path.join(
                WORKFLOW_UPLOAD_FOLDER,
                f"{request.session['heatmap_session_id']}_heatmap_data.csv"
            )) if "heatmap_session_id" in request.session else False
        }
    )

@app.post("/workflows/heatmap/upload")
async def upload_heatmap_file(request: Request, file: UploadFile = File(...)):
    # Initialize session if needed
    if "heatmap_session_id" not in request.session:
        request.session["heatmap_session_id"] = str(uuid.uuid4())
    
    session_id = request.session["heatmap_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_heatmap_data.csv")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify the file is a valid CSV
        df = pd.read_csv(file_path)
        
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "preview": df.head().to_html(classes='table table-striped'),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/heatmap/generate")
async def generate_heatmap(request: Request, x_axis: str = Form(...), y_axis: str = Form(...), value: str = Form(...), color_scheme: str = Form(...)):
    if "heatmap_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["heatmap_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_heatmap_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No file uploaded"
        }, status_code=400)
    
    try:
        df = pd.read_csv(file_path)
        pivot_table = df.pivot_table(index=y_axis, columns=x_axis, values=value)
        
        # Generate heatmap using seaborn
        import seaborn as sns
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, cmap=color_scheme, annot=True, fmt=".2f")
        plt.title(f"Heatmap of {value} by {x_axis} and {y_axis}")
        
        # Save plot to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        # Convert to base64
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "heatmap": f'<img src="data:image/png;base64,{image_base64}" alt="Heatmap">'
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/workflows/eda")
async def eda_page(request: Request):
    # Initialize session if needed
    if "eda_session_id" not in request.session:
        request.session["eda_session_id"] = str(uuid.uuid4())
    
    return templates.TemplateResponse(
        "eda.html",
        {
            "request": request,
            "title": "Drishti.AI - Exploratory Data Analysis",
            "active_page": "workflows",
            "has_file": os.path.exists(os.path.join(
                WORKFLOW_UPLOAD_FOLDER,
                f"{request.session['eda_session_id']}_eda_data.csv"
            )) if "eda_session_id" in request.session else False
        }
    )

@app.post("/workflows/eda/upload")
async def upload_eda_file(request: Request, file: UploadFile = File(...)):
    # Initialize session if needed
    if "eda_session_id" not in request.session:
        request.session["eda_session_id"] = str(uuid.uuid4())
    
    session_id = request.session["eda_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_eda_data.csv")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify the file is a valid CSV
        df = pd.read_csv(file_path)
        
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "preview": df.head().to_html(classes='table table-striped'),
            "columns": df.columns.tolist(),
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isna().sum().sum()
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/eda/analyze-column")
async def analyze_column(request: Request, column: str = Form(...)):
    if "eda_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["eda_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_eda_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No file uploaded"
        }, status_code=400)
    
    try:
        df = pd.read_csv(file_path)
        column_data = df[column]
        
        # Generate statistics
        stats = {
            "count": len(column_data),
            "mean": column_data.mean(),
            "std": column_data.std(),
            "min": column_data.min(),
            "25%": column_data.quantile(0.25),
            "50%": column_data.quantile(0.5),
            "75%": column_data.quantile(0.75),
            "max": column_data.max(),
            "missing": column_data.isna().sum()
        }
        
        # Generate histogram
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        plt.figure(figsize=(10, 6))
        plt.hist(column_data.dropna(), bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "results": f"""
                <h3>Statistics for {column}</h3>
                <table class="table table-striped">
                    <tr><th>Count</th><td>{stats['count']}</td></tr>
                    <tr><th>Mean</th><td>{stats['mean']:.2f}</td></tr>
                    <tr><th>Standard Deviation</th><td>{stats['std']:.2f}</td></tr>
                    <tr><th>Minimum</th><td>{stats['min']:.2f}</td></tr>
                    <tr><th>25th Percentile</th><td>{stats['25%']:.2f}</td></tr>
                    <tr><th>Median</th><td>{stats['50%']:.2f}</td></tr>
                    <tr><th>75th Percentile</th><td>{stats['75%']:.2f}</td></tr>
                    <tr><th>Maximum</th><td>{stats['max']:.2f}</td></tr>
                    <tr><th>Missing Values</th><td>{stats['missing']}</td></tr>
                </table>
                <img src="data:image/png;base64,{image_base64}" alt="Histogram">
            """
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/eda/analyze-correlation")
async def analyze_correlation(request: Request, columns: str = Form(...)):
    if "eda_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["eda_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_eda_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No file uploaded"
        }, status_code=400)
    
    try:
        df = pd.read_csv(file_path)
        selected_columns = json.loads(columns)
        
        # Calculate correlation matrix
        corr_matrix = df[selected_columns].corr()
        
        # Generate correlation heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        import base64
        from io import BytesIO
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title("Correlation Matrix")
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "results": f"""
                <h3>Correlation Matrix</h3>
                <img src="data:image/png;base64,{image_base64}" alt="Correlation Matrix">
                <h3>Correlation Values</h3>
                {corr_matrix.to_html(classes='table table-striped')}
            """
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.get("/workflows/clustering")
async def clustering_page(request: Request):
    # Initialize session if needed
    if "clustering_session_id" not in request.session:
        request.session["clustering_session_id"] = str(uuid.uuid4())
    
    return templates.TemplateResponse(
        "clustering.html",
        {
            "request": request,
            "title": "Drishti.AI - Clustering Analysis",
            "active_page": "workflows",
            "has_file": os.path.exists(os.path.join(
                WORKFLOW_UPLOAD_FOLDER,
                f"{request.session['clustering_session_id']}_clustering_data.csv"
            )) if "clustering_session_id" in request.session else False
        }
    )

@app.post("/workflows/clustering/upload")
async def upload_clustering_file(request: Request, file: UploadFile = File(...)):
    # Initialize session if needed
    if "clustering_session_id" not in request.session:
        request.session["clustering_session_id"] = str(uuid.uuid4())
    
    session_id = request.session["clustering_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_clustering_data.csv")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify the file is a valid CSV
        df = pd.read_csv(file_path)
        
        return JSONResponse({
            "status": "success",
            "message": "File uploaded successfully",
            "preview": df.head().to_html(classes='table table-striped'),
            "columns": df.columns.tolist()
        })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

@app.post("/workflows/clustering/perform")
async def perform_clustering(request: Request, algorithm: str = Form(...), n_clusters: int = Form(...), visualization: str = Form(...), features: str = Form(...)):
    if "clustering_session_id" not in request.session:
        return JSONResponse({
            "status": "error",
            "message": "No active session"
        }, status_code=400)
    
    session_id = request.session["clustering_session_id"]
    file_path = os.path.join(WORKFLOW_UPLOAD_FOLDER, f"{session_id}_clustering_data.csv")
    
    if not os.path.exists(file_path):
        return JSONResponse({
            "status": "error",
            "message": "No file uploaded"
        }, status_code=400)
    
    try:
        df = pd.read_csv(file_path)
        selected_features = json.loads(features)
        
        # Ensure all selected features are numeric
        for feature in selected_features:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=selected_features)
        
        if len(df) == 0:
            return JSONResponse({
                "status": "error",
                "message": "No valid numeric data found in selected features"
            }, status_code=400)
        
        X = df[selected_features]
        
        # Perform clustering based on selected algorithm
        if algorithm == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif algorithm == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        else:  # hierarchical
            model = AgglomerativeClustering(n_clusters=n_clusters)
        
        clusters = model.fit_predict(X)
        
        # Generate visualization
        plt.figure(figsize=(10, 8))
        
        if visualization == "scatter":
            if len(selected_features) >= 2:
                plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters, cmap='viridis')
                plt.title(f"{algorithm.upper()} Clustering Results")
                plt.xlabel(selected_features[0])
                plt.ylabel(selected_features[1])
            else:
                raise ValueError("Need at least 2 features for scatter plot")
        else:  # dendrogram
            if algorithm == "hierarchical":
                from scipy.cluster.hierarchy import dendrogram, linkage
                Z = linkage(X, 'ward')
                dendrogram(Z)
                plt.title("Hierarchical Clustering Dendrogram")
            else:
                raise ValueError("Dendrogram is only available for hierarchical clustering")
        
        # Save plot to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_png).decode('utf-8')
        
        # Generate cluster statistics
        df['cluster'] = clusters
        # Calculate statistics for numeric columns only
        numeric_stats = df.groupby('cluster')[selected_features].agg(['mean', 'std', 'count']).round(2)
        
        return JSONResponse({
            "status": "success",
            "visualization": f'<img src="data:image/png;base64,{image_base64}" alt="Clustering Results">',
            "results": f"""
                <h3>Cluster Statistics</h3>
                {numeric_stats.to_html(classes='table table-striped')}
            """
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
