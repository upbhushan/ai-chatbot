from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create FastAPI app FIRST
app = FastAPI()

# 🌍 Enable CORS (very important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq model
chat_model = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# LangGraph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state["messages"]
    response = chat_model.invoke(messages)
    return {"messages": [response]}

# 🏗 Build graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()

# Request model
class ChatRequest(BaseModel):
    message: str

#  Chat endpoint
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    initial_state = {
        "messages": [HumanMessage(content=request.message)]
    }

    result = chatbot.invoke(initial_state)

    return {
        "response": result["messages"][-1].content
    }
