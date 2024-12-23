from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def create_conversation_chain():
    """Create a simple conversation chain with Turkish language support"""
    
    # Create the language model
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Always respond in Turkish. Be friendly and conversational."),
        ("human", "{input}")
    ])
    
    # Create the chain
    chain = prompt | llm
    
    def invoke_chain(message: str, session_id: str = "default") -> str:
        """Helper function to invoke the chain"""
        return chain.invoke({"input": message})
    
    return invoke_chain