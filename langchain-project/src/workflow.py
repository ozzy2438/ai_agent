from typing import TypedDict, Annotated, Sequence, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Define state schema
class WorkflowState(TypedDict):
    messages: Sequence[HumanMessage | AIMessage]
    next: str

def research(state: WorkflowState) -> WorkflowState:
    """Research node that processes the input and generates a response"""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [], "next": END}
    
    # Get the last message
    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"messages": messages, "next": END}
    
    # Process with LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    ai_message = llm.invoke([last_message])
    
    return {
        "messages": messages + [ai_message],
        "next": END
    }

def create_workflow() -> StateGraph:
    """Creates a workflow graph with proper state management"""
    
    # Initialize workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("research", research)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add conditional edges
    workflow.add_edge("research", END)
    
    # Compile workflow
    workflow = workflow.compile()
    
    def invoke_workflow(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Helper function to invoke the workflow with proper message format"""
        if isinstance(input_data, dict) and "messages" in input_data:
            # Convert message dict to HumanMessage
            if isinstance(input_data["messages"], list):
                messages = []
                for msg in input_data["messages"]:
                    if isinstance(msg, dict) and "content" in msg:
                        messages.append(HumanMessage(content=msg["content"]))
                return workflow.invoke({"messages": messages})
        
        # Default case: create a single message
        return workflow.invoke({
            "messages": [HumanMessage(content=str(input_data.get("input", "")))]
        })
    
    return invoke_workflow 