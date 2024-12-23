# Local Development Setup Guide

## 1. Environment Setup

### Prerequisites
- Python 3.11
- OpenAI API key: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
- SERPAPI_API_KEY:f6a5fad91ad00a424888561941d8232426fcedf2bb3af3f30ae19577a3c14a45

### Initial Setup

1. Create a new virtual environment:
```bash
python -m venv langchain-env
```

2. Activate the virtual environment:

```
- Unix/MacOS:
```bash
source langchain-env/bin/activate
```

1. Install required packages:
```bash
pip install langchain openai chromadb tiktoken langgraph
pip install unstructured pypdf python-magic-bin
```

## 2. Project Structure

Create the following directory structure:
```
langchain-project/
├── .env                  # Environment variables
├── data/                 # Your document files
├── src/
│   ├── __init__.py
│   ├── rag.py           # RAG implementation
│   ├── agents.py        # Custom agents
│   ├── memory.py        # Memory management
│   └── workflow.py      # LangGraph workflow
├── tests/               # Test files
└── main.py             # Main application entry
```

## 3. Environment Configuration

Create a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
SERPAPI_API_KEY=your_serpapi_key_here  # If using web search
```

## 4. Running the Project

1. Test RAG Implementation:
```python
from src.rag import rag_pipeline

response = rag_pipeline("./data", "What are the key points in the documents?")
print(response)
```

2. Test Conversation Chain:
```python
from src.memory import setup_conversation_chain

conversation = setup_conversation_chain()
response = conversation.predict(input="Hello!")
print(response)
```

3. Test Complete Workflow:
```python
from src.workflow import create_workflow

workflow = create_workflow()
response = workflow.run({"messages": [], "current_step": "research"})
print(response)
```

## 5. Common Issues and Solutions

1. **ChromaDB Issues**
- If you encounter ChromaDB errors, try:
```bash
pip uninstall chromadb
pip install chromadb --force-reinstall
```

2. **OpenAI API Rate Limits**
- Implement exponential backoff:
```python
from tenacity import retry, wait_exponential
@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def rate_limited_completion(*args, **kwargs):
    # Your OpenAI API call here
    pass
```

3. **Memory Issues with Large Documents**
- Use batch processing:
```python
def process_in_batches(documents, batch_size=5):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        # Process batch
        yield process_batch(batch)
```

## 6. Testing

Create basic tests:

```python
# tests/test_rag.py
import unittest
from src.rag import rag_pipeline

class TestRAG(unittest.TestCase):
    def test_rag_pipeline(self):
        response = rag_pipeline("./test_data", "test query")
        self.assertIsNotNone(response)
```

Run tests:
```bash
python -m unittest discover tests
```

## 7. Performance Optimization

1. **Embedding Caching**
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_embedding(text: str) -> List[float]:
    return embeddings.embed_query(text)
```

2. **Batch Processing**
```python
def batch_process_documents(documents: List[Document], batch_size: int = 5):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        yield process_batch(batch)
```

## 8. Monitoring and Logging

Add logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
```

## 9. Next Steps

1. Implement error handling and retry mechanisms
2. Add API endpoints using FastAPI or Flask
3. Set up monitoring and logging
4. Add authentication and rate limiting
5. Implement caching strategies
6. Add documentation using Sphinx or MkDocs

## 10. Production Considerations

1. Use environment variables for all sensitive data
2. Implement proper error handling and logging
3. Set up monitoring and alerting
4. Consider scaling solutions (Redis, Kubernetes, etc.)
5. Implement security best practices
6. Set up CI/CD pipelines