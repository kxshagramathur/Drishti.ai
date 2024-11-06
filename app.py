import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import re
import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

if os.getenv("GROQ_API_KEY") is None or os.getenv("GROQ_API_KEY") == "":
    st.write("GROQ_API_KEY is not set")
    exit(1)

TEMP_FOLDER = os.path.join(os.getcwd(), "temp_uploads")

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def csv_agent_func(file_path, user_message):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
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
        st.write(f"Error: {e}")
        return None

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

def csv_analyzer_app():
    st.title('Drishti.ai (Prototype 1)')
    st.write('Please upload your CSV file and enter your query below:')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
        st.write(file_details)
        
        file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        df = pd.read_csv(file_path)
        st.dataframe(df)
        
        user_input = st.text_input("Your query")
        if st.button('Run'):
            with st.spinner('Processing your query...'):
                response = csv_agent_func(file_path, user_input)
                
                if response:
                    if has_visualization_request(user_input):
                        code_to_execute = extract_code_from_response(response)
                        if code_to_execute:
                            try:
                                fig = execute_visualization(code_to_execute, df)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error in visualization: {str(e)}")
                    else:
                        st.write("Response:")
                        st.write(response)

if __name__ == '__main__':
    csv_analyzer_app()