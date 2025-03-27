from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, send_file
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv
import uuid
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.secret_key = 'your_secret_key_here'  # Required for session management

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def csv_agent_func(file_path, user_message):
    llm = ChatCerebras(model="llama3.3-70b", temperature=0)
    agent = create_csv_agent(
        llm,
        file_path, 
        verbose=True,
        allow_dangerous_code=True
    )
    try:
        response = agent.invoke(user_message)
        return str(response['output'])
    except Exception as e:
        return f"Error: {e}"

def extract_code_from_response(response):
    if not isinstance(response, str):
        return None
    code_pattern = r"```python\s*(.*?)\s*```"
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        code = code.replace('\r', '\n')
        code = '\n'.join(line.strip() for line in code.split('\n'))
        return code
    return None

def execute_visualization(code, df):
    try:
        plt.clf()
        plt.figure(figsize=(10, 6))
        local_vars = {'df': df, 'plt': plt}
        exec(code, globals(), local_vars)
        plt.tight_layout()
        return plt.gcf()
    except Exception as e:
        raise Exception(f"Visualization error: {str(e)}")

def has_visualization_request(response):
    visualization_keywords = ['plot', 'chart', 'graph', 'visualize', 'visualization', 'figure']
    return any(keyword in response.lower() for keyword in visualization_keywords)

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    # Define file paths based on session ID
    session_id = session['session_id']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_data.csv")

    if request.method == 'POST':
        if 'reset' in request.form:
            # Reset the session and remove the file
            if os.path.exists(file_path):
                os.remove(file_path)
            session.pop('has_file', None)
            session.pop('original_filename', None)
            session['session_id'] = str(uuid.uuid4())
            return redirect(url_for('index'))

        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file.filename != '':
                # Save file with session ID
                file.save(file_path)
                session['has_file'] = True
                session['original_filename'] = file.filename
                return redirect(url_for('index'))

        if 'query' in request.form:
            # Handle query submission
            if os.path.exists(file_path) and session.get('has_file', False):
                user_input = request.form.get('query')
                response = csv_agent_func(file_path, user_input)

                if has_visualization_request(user_input):
                    code_to_execute = extract_code_from_response(response)
                    if code_to_execute:
                        try:
                            df = pd.read_csv(file_path)
                            fig = execute_visualization(code_to_execute, df)
                            plot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_plot.png")
                            fig.savefig(plot_path)
                            return render_template('index.html', 
                                                   response=response, 
                                                   plot_image=f"{session_id}_plot.png",
                                                   filename=session.get('original_filename', 'Uploaded file'))
                        except Exception as e:
                            return render_template('index.html', 
                                                   error=f"Error in visualization: {str(e)}",
                                                   filename=session.get('original_filename', 'Uploaded file'))
                return render_template('index.html', 
                                       response=response,
                                       filename=session.get('original_filename', 'Uploaded file'))
            else:
                return render_template('index.html', error="Please upload a CSV file first.")

    # For GET requests or after processing POST
    if os.path.exists(file_path) and session.get('has_file', False):
        return render_template('index.html', filename=session.get('original_filename', 'Uploaded file'))
    else:
        return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/workflows', methods=['GET', 'POST'])
def workflows():
    if 'workflow_session_id' not in session:
        session['workflow_session_id'] = str(uuid.uuid4())
        session['workflow_history'] = []
    
    session_id = session['workflow_session_id']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_workflow_data.csv")

    if request.method == 'POST':
        if 'workflow_file' in request.files:
            file = request.files['workflow_file']
            if file.filename != '':
                file.save(file_path)
                session['has_workflow_file'] = True
                session['workflow_filename'] = file.filename
                session['workflow_history'] = []  # Reset history for new file
                return redirect(url_for('workflows'))

        if 'action' in request.form:
            if os.path.exists(file_path) and session.get('has_workflow_file', False):
                action = request.form.get('action')
                df = pd.read_csv(file_path)
                initial_rows = int(len(df))
                initial_cols = int(len(df.columns))
                
                if action == 'drop_rows':
                    df.dropna(inplace=True)
                    affected_rows = int(initial_rows - len(df))
                    affected_cols = 0
                    action_desc = "Dropped rows with missing values"
                elif action == 'replace_mean':
                    missing_counts = df.isna().sum()
                    df.fillna(df.mean(numeric_only=True), inplace=True)
                    affected_rows = int(missing_counts.sum())
                    affected_cols = 0
                    action_desc = "Replaced missing values with mean"
                elif action == 'remove_duplicates':
                    initial_duplicates = int(df.duplicated().sum())
                    df.drop_duplicates(inplace=True)
                    affected_rows = initial_duplicates
                    affected_cols = 0
                    action_desc = "Removed duplicate rows"
                elif action == 'remove_empty_columns':
                    initial_empty_cols = int(df.isna().all().sum())
                    df.dropna(axis=1, how='all', inplace=True)
                    affected_rows = 0
                    affected_cols = initial_empty_cols
                    action_desc = "Removed empty columns"
                
                df.to_csv(file_path, index=False)
                
                # Add to workflow history with Python native types
                history_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': action_desc,
                    'initial_rows': initial_rows,
                    'initial_cols': initial_cols,
                    'affected_rows': affected_rows,
                    'affected_cols': affected_cols,
                    'final_rows': int(len(df)),
                    'final_cols': int(len(df.columns))
                }
                session['workflow_history'].append(history_entry)
                session.modified = True  # Ensure session is saved
                
                return redirect(url_for('workflows'))

    return render_template('workflows.html', 
                         filename=session.get('workflow_filename'),
                         has_file=session.get('has_workflow_file', False),
                         history=session.get('workflow_history', []))

@app.route('/download_workflow_file')
def download_workflow_file():
    if 'workflow_session_id' in session:
        session_id = session['workflow_session_id']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_workflow_data.csv")
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
    return "No file available for download", 404

if __name__ == '__main__':
    app.run(debug=True)