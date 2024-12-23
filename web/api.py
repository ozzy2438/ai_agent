import sys
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from pydantic import BaseModel
from datetime import datetime
from web.db import Database
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.callbacks import get_openai_callback
from langchain_community.utilities import SerpAPIWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    description="API for document processing and question answering using LangChain",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
search = SerpAPIWrapper()

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Chroma client
vectorstore = None

def get_vectorstore(collection_name: str = "default") -> Chroma:
    """Get or create vectorstore for collection"""
    global vectorstore
    if vectorstore is None or vectorstore._collection_name != collection_name:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=os.path.join(DATA_DIR, "chroma")
        )
    return vectorstore

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = "default"
    k: Optional[int] = 3

class URLRequest(BaseModel):
    urls: List[str]
    collection_name: Optional[str] = "default"

@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    collection_name: str = Query("default", description="Name of the collection to store documents")
):
    """Upload and process documents"""
    try:
        # Save uploaded files
        saved_files = []
        for file in files:
            file_path = os.path.join(DATA_DIR, file.filename)
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(file_path)
            
            # Save to Supabase
            await Database.save_document(
                filename=file.filename,
                content=content.decode(),
                metadata={"path": file_path},
                collection_name=collection_name
            )
        
        # Load and process documents
        documents = []
        for file_path in saved_files:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        # Split documents
        texts = text_splitter.split_documents(documents)
        
        # Create or update vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=collection_name
        )
        
        return {
            "message": f"Successfully processed {len(files)} files",
            "collection": collection_name,
            "files": [file.filename for file in files]
        }
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-urls")
async def load_urls(request: URLRequest):
    """Load and process documents from URLs"""
    try:
        # Load documents from URLs
        loader = UnstructuredURLLoader(urls=request.urls)
        documents = loader.load()
        
        # Save to Supabase
        for i, doc in enumerate(documents):
            await Database.save_document(
                filename=f"url_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                content=doc.page_content,
                metadata=doc.metadata,
                collection_name=request.collection_name
            )
        
        # Split documents
        texts = text_splitter.split_documents(documents)
        
        # Create or update vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            collection_name=request.collection_name
        )
        
        return {
            "message": f"Successfully processed {len(request.urls)} URLs",
            "collection": request.collection_name,
            "urls": request.urls
        }
    except Exception as e:
        logger.error(f"Error processing URLs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """Query the RAG pipeline"""
    try:
        # Get vector store
        vectorstore = Chroma(
            collection_name=request.collection_name,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": request.k}
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Query with cost tracking
        with get_openai_callback() as cb:
            response = qa_chain(request.query)
            
            result = {
                "result": response["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in response["source_documents"]
                ],
                "cost": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            }
        
        # Save query to Supabase
        await Database.save_query(
            query=request.query,
            answer=result["result"],
            source_documents=result["source_documents"],
            token_usage=result["cost"],
            total_cost=result["cost"]["total_cost"]
        )
        
        return result
    except Exception as e:
        logger.error(f"Error querying RAG pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List available collections"""
    try:
        vectorstore = Chroma(embedding_function=embeddings)
        collections = vectorstore._client.list_collections()
        return {"collections": [col.name for col in collections]}
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    """Get analytics data"""
    try:
        return await Database.get_analytics()
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query-history")
async def get_query_history():
    """Get query history"""
    try:
        return await Database.get_query_history()
    except Exception as e:
        logger.error(f"Error getting query history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 