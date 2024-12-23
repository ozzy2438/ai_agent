from typing import List, Optional, Dict, Any
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader, 
    UnstructuredURLLoader, 
    PyPDFLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader,
    JSONLoader
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.callbacks.manager import get_openai_callback
import magic
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.vectorstore = None
        self.supported_formats = {
            'text/plain': TextLoader,
            'application/pdf': PyPDFLoader,
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': UnstructuredPowerPointLoader,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': UnstructuredWordDocumentLoader,
            'text/csv': CSVLoader,
            'application/json': JSONLoader
        }
        
    def _get_file_type(self, file_path: str) -> str:
        """Detect file type using python-magic"""
        try:
            return magic.from_file(file_path, mime=True)
        except Exception as e:
            logger.error(f"Error detecting file type for {file_path}: {str(e)}")
            return None
        
    def _get_appropriate_loader(self, file_path: str):
        """Get appropriate loader based on file type"""
        file_type = self._get_file_type(file_path)
        if file_type in self.supported_formats:
            return self.supported_formats[file_type]
        else:
            logger.warning(f"Unsupported file type {file_type} for {file_path}")
            return None
        
    def load_documents(self, file_pattern: str = "**/*.*") -> List:
        """Load documents from directory with enhanced format support"""
        documents = []
        
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                loader_class = self._get_appropriate_loader(file_path)
                
                if loader_class:
                    try:
                        logger.info(f"Loading document: {file_path}")
                        loader = loader_class(file_path)
                        documents.extend(loader.load())
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {str(e)}")
                        
        return documents
    
    def load_from_url(self, urls: List[str], retry_count: int = 3) -> List:
        """Load documents from URLs with retry mechanism"""
        documents = []
        
        for url in urls:
            for attempt in range(retry_count):
                try:
                    logger.info(f"Loading content from URL: {url} (attempt {attempt + 1})")
                    loader = UnstructuredURLLoader(urls=[url])
                    documents.extend(loader.load())
                    break
                except Exception as e:
                    if attempt == retry_count - 1:
                        logger.error(f"Failed to load URL {url} after {retry_count} attempts: {str(e)}")
                    else:
                        logger.warning(f"Attempt {attempt + 1} failed for URL {url}: {str(e)}")
                        
        return documents
    
    def process_documents(self, documents: List, chunk_size: int = 1000, content_type: str = "text") -> List:
        """Process and split documents with content-aware splitting"""
        splitter_map = {
            "text": RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,
                length_function=len
            ),
            "markdown": MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200
            ),
            "code": PythonCodeTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200
            )
        }
        
        splitter = splitter_map.get(content_type, splitter_map["text"])
        
        try:
            logger.info(f"Processing documents with {content_type} splitter")
            return splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []
    
    def create_vectorstore(self, documents: Optional[List] = None, collection_name: str = "default"):
        """Create or update vector store with collection support"""
        try:
            if documents is None:
                documents = self.process_documents(self.load_documents())
                
            if self.vectorstore is None:
                logger.info(f"Creating new vector store collection: {collection_name}")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    collection_name=collection_name
                )
            else:
                logger.info(f"Adding documents to existing vector store collection: {collection_name}")
                self.vectorstore.add_documents(documents)
        except Exception as e:
            logger.error(f"Error creating/updating vector store: {str(e)}")
            raise
    
    def query(self, query: str, k: int = 3, collection_name: str = "default") -> Dict[str, Any]:
        """Query the RAG pipeline with enhanced features"""
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Please load documents first.")
            
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                    "filter": {"collection": collection_name} if collection_name != "default" else None
                }
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            with get_openai_callback() as cb:
                response = qa_chain.invoke({"query": query})
                
            # Extract source document metadata
            sources = []
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content[:200] + "...",  # First 200 chars
                    "metadata": doc.metadata
                })
                
            return {
                "result": response["result"],
                "source_documents": sources,
                "cost": {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost
                }
            }
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {str(e)}")
            raise

def rag_pipeline(data_dir: str, query: str):
    """Legacy support for old interface"""
    rag = RAGPipeline(data_dir)
    documents = rag.load_documents()
    splits = rag.process_documents(documents)
    rag.create_vectorstore(splits)
    return rag.query(query)["result"] 