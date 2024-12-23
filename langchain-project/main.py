import os
import logging
from dotenv import load_dotenv
from src.memory import create_conversation_chain
from src.workflow import create_workflow
from src.rag import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_rag_pipeline():
    """Test the RAG pipeline functionality"""
    try:
        logger.info("Testing RAG pipeline...")
        rag = RAGPipeline(data_dir="data")
        
        # Load and process documents
        docs = rag.load_documents()
        logger.info(f"Loaded {len(docs)} documents")
        
        # Create vector store
        rag.create_vectorstore()
        
        # Test query with cost tracking
        question = "What are the key points about LangChain?"
        response = rag.query(question)
        logger.info(f"\nQuestion: {question}")
        logger.info(f"Answer: {response['result']}")
        logger.info(f"Total Cost: ${response['cost']['total_cost']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return False

def test_conversation_chain():
    """Test the conversation chain functionality"""
    try:
        logger.info("\nTesting conversation chain...")
        chain = create_conversation_chain()
        response = chain("Merhaba!", "test-session")
        logger.info(f"Bot: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error in conversation chain: {str(e)}")
        return False

def test_workflow():
    """Test the workflow functionality"""
    try:
        logger.info("\nTesting workflow...")
        workflow = create_workflow()
        response = workflow({"input": "Tell me about LangChain"})
        logger.info(f"Workflow response: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        return False

def main():
    """Main function to run all tests"""
    try:
        # Check for required environment variables
        required_vars = ['OPENAI_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Run all tests
        rag_success = test_rag_pipeline()
        conversation_success = test_conversation_chain()
        workflow_success = test_workflow()
        
        # Report overall status
        if all([rag_success, conversation_success, workflow_success]):
            logger.info("\nAll tests completed successfully!")
        else:
            logger.warning("\nSome tests failed. Check the logs for details.")
            
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 