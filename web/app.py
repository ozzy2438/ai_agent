import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
from datetime import datetime
import asyncio
import sys
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web.db import Database

# Configure API endpoint and timeout
API_URL = "http://localhost:8000"
TIMEOUT = 30  # seconds

# Configure requests session with retry
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def wait_for_api(max_retries: int = 5, delay: int = 2):
    """Wait for API to become available"""
    for i in range(max_retries):
        try:
            response = session.get(f"{API_URL}/collections", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if i < max_retries - 1:
                time.sleep(delay)
                continue
            st.error("API sunucusuna bağlanılamıyor. Lütfen API'nin çalıştığından emin olun.")
            return False
    return False

# Wait for API at startup
if not wait_for_api():
    st.stop()

# Set page config
st.set_page_config(
    page_title="LangChain RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    .stats-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Add session state initialization at the beginning
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

async def login(email: str, password: str) -> bool:
    """Handle login process using Supabase Auth"""
    user = await Database.sign_in_with_email(email, password)
    if user:
        st.session_state.authenticated = True
        st.session_state.user = user
        return True
    return False

# Login page if not authenticated
if not st.session_state.authenticated:
    st.title("🔐 Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if asyncio.run(login(email, password)):
                st.success("Successfully logged in!")
                st.experimental_rerun()
            else:
                st.error("Invalid email or password")
    st.stop()

# Add user info in sidebar
st.sidebar.title("🤖 LangChain RAG System")
if st.session_state.user:
    st.sidebar.text(f"Logged in as: {st.session_state.user.email}")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.experimental_rerun()

def get_collections() -> List[str]:
    """Get list of available collections"""
    try:
        response = session.get(f"{API_URL}/collections", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()["collections"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching collections: {str(e)}")
        return ["default"]

def upload_files(files: List[Any], collection_name: str):
    """Upload files to the API"""
    try:
        files_data = [("files", file) for file in files]
        response = session.post(
            f"{API_URL}/upload",
            files=files_data,
            params={"collection_name": collection_name},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading files: {str(e)}")
        return None

def load_urls(urls: List[str], collection_name: str):
    """Load documents from URLs"""
    try:
        response = session.post(
            f"{API_URL}/load-urls",
            json={"urls": urls, "collection_name": collection_name},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading URLs: {str(e)}")
        return None

def query_rag(query: str, collection_name: str, k: int = 3):
    """Query the RAG system"""
    try:
        response = session.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "collection_name": collection_name,
                "k": k
            },
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error querying RAG system: {str(e)}")
        return None

def get_analytics():
    """Get analytics data"""
    try:
        response = session.get(f"{API_URL}/analytics", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching analytics: {str(e)}")
        return None

def get_query_history():
    """Get query history"""
    try:
        response = session.get(f"{API_URL}/query-history", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching query history: {str(e)}")
        return None

# Sidebar
page = st.sidebar.selectbox(
    "Choose a page",
    ["Document Management", "Query System", "Analytics"]
)

# Main content
if page == "Document Management":
    st.title("📚 Document Management")
    
    # Collection selection
    collections = get_collections()
    collection_name = st.selectbox("Select Collection", collections)
    
    # File upload
    st.header("Upload Files")
    with st.expander("Upload Files", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            accept_multiple_files=True,
            type=["txt", "pdf", "docx", "pptx", "csv", "json"]
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    result = upload_files(uploaded_files, collection_name)
                    if result:
                        st.success(result["message"])
                        st.json(result)
    
    # URL input
    st.header("Load from URLs")
    with st.expander("Load URLs", expanded=True):
        urls_text = st.text_area(
            "Enter URLs (one per line)",
            height=150,
            help="Enter URLs of web pages to process"
        )
        
        if urls_text and st.button("Process URLs"):
            urls = [url.strip() for url in urls_text.split("\n") if url.strip()]
            with st.spinner("Processing URLs..."):
                result = load_urls(urls, collection_name)
                if result:
                    st.success(result["message"])
                    st.json(result)

elif page == "Query System":
    st.title("🔍 Query System")
    
    # Collection selection
    collections = get_collections()
    collection_name = st.selectbox("Select Collection", collections)
    
    # Query input
    query = st.text_area("Enter your question", height=100)
    k = st.slider("Number of relevant documents", min_value=1, max_value=5, value=3)
    
    if query and st.button("Submit Query"):
        with st.spinner("Processing query..."):
            result = query_rag(query, collection_name, k)
            if result:
                # Display answer
                st.header("Answer")
                st.write(result["result"])
                
                # Display source documents
                st.header("Source Documents")
                for i, doc in enumerate(result["source_documents"], 1):
                    with st.expander(f"Document {i}"):
                        st.write(doc["content"])
                        st.json(doc["metadata"])
                
                # Display cost information
                st.header("Cost Information")
                cost_df = pd.DataFrame({
                    "Metric": ["Total Tokens", "Prompt Tokens", "Completion Tokens"],
                    "Value": [
                        result["cost"]["total_tokens"],
                        result["cost"]["prompt_tokens"],
                        result["cost"]["completion_tokens"]
                    ]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(cost_df)
                with col2:
                    fig = px.pie(cost_df, values="Value", names="Metric", title="Token Usage")
                    st.plotly_chart(fig)
                
                st.info(f"Total Cost: ${result['cost']['total_cost']:.4f}")
    
    # Query history
    st.header("Query History")
    history = get_query_history()
    if history:
        for query in history:
            with st.expander(f"Query: {query['query'][:100]}..."):
                st.write("Answer:", query["answer"])
                st.write("Cost:", f"${query['total_cost']:.4f}")
                st.write("Time:", query["created_at"])

else:  # Analytics page
    st.title("📊 Analytics")
    
    analytics = get_analytics()
    if analytics:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", analytics["total_queries"])
        with col2:
            st.metric("Total Cost", f"${analytics['total_cost']:.2f}")
        with col3:
            st.metric("Average Tokens", f"{analytics['average_tokens']:.0f}")
        
        # Query trends
        st.header("Query Trends")
        queries_df = pd.DataFrame(analytics["queries_by_date"])
        queries_df["created_at"] = pd.to_datetime(queries_df["created_at"])
        
        # Daily query count
        daily_queries = queries_df.groupby(queries_df["created_at"].dt.date).size().reset_index()
        daily_queries.columns = ["date", "count"]
        
        fig = px.line(daily_queries, x="date", y="count", title="Daily Queries")
        st.plotly_chart(fig)
        
        # Cost trends
        st.header("Cost Trends")
        daily_cost = queries_df.groupby(queries_df["created_at"].dt.date)["total_cost"].sum().reset_index()
        
        fig = px.line(daily_cost, x="created_at", y="total_cost", title="Daily Cost")
        st.plotly_chart(fig)
        
        # Token usage distribution
        st.header("Token Usage Distribution")
        token_usage = pd.DataFrame([
            json.loads(query["token_usage"]) for query in analytics["queries_by_date"]
        ])
        
        fig = px.box(token_usage, title="Token Usage Distribution")
        st.plotly_chart(fig)
    else:
        st.info("No analytics data available yet. Start using the system to generate analytics.")