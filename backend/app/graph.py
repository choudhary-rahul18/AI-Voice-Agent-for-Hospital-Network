# backend/app/graph.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from app.tools import loop_tools # Imported from the previous step
import logging

# Configure the logger to show timestamps and severity levels
logging.basicConfig(
    level=logging.INFO, # Change to DEBUG to see more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 1. Define the State (Tracks conversation history)
class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph():
    # 2. Initialize Gemini (requires GEMINI_API_KEY in .env)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # Bind our SQL search tool to the model
    llm_with_tools = llm.bind_tools(loop_tools)

    # 3. Define the Core Agent Node
    def agent_node(state: State):
        is_first_turn = len(state["messages"]) <= 1
        
        if is_first_turn:
            greeting_rule = "Introduce yourself as 'Loop AI' since this is the beginning of the conversation."
        else:
            greeting_rule = "DO NOT introduce yourself. You are in the middle of a conversation. Speak naturally and directly answer the follow-up."

        # 3. THE UPGRADED SYSTEM PROMPT
        system_message = SystemMessage(content=f"""
        You are 'Loop AI', a helpful voice assistant for a hospital network.
        
        {greeting_rule}

        SCOPE & INTENT RULES:
        - You are fully authorized to provide hospital names, addresses, and confirm if they are in network.
        - SPEECH TYPOS: If the user asks for something that sounds phonetically similar to hospital (like "hostel"), assume the microphone misheard them and answer regarding hospitals anyway, or ask a clarifying question. DO NOT reject them.
        
        TOOL RULES:
        - If the user asks "How many" hospitals there are, you MUST use the `count_hospitals` tool.
        - If the user asks to "list" or "find" hospitals, use the `search_hospitals` tool. Use the 'offset' parameter if they ask for "more".
        - Use your memory to answer follow-up questions about addresses if you already searched for them.

        OUT OF SCOPE RULES:
        - ONLY reject questions that are 100% unrelated to healthcare, hospitals, or addresses (e.g., coding, weather, recipes). 
        - If truly out of scope, reply exactly with: "I'm sorry, I can't help with that. I am forwarding this to a human agent."
        """)
        
        messages_to_send = [system_message] + state["messages"]
        response = llm_with_tools.invoke(messages_to_send)
        
        # 4. Update logging to use the logger you set up
        if response.tool_calls:
            logger.info(f"\n[SYSTEM LOG] 🧠 AGENT DECISION: Calling Tools: {[t['name'] for t in response.tool_calls]}")
        else:
            logger.info("\n[SYSTEM LOG] 🧠 AGENT DECISION: Answered from Memory/Context.")

        return {"messages": [response]}

    # 4. Construct the Graph
    workflow = StateGraph(State)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(loop_tools))
    
    # 5. Define Routing Logic
    workflow.add_edge(START, "agent")
    # If Gemini decides to use a tool, go to tools. Otherwise, end.
    workflow.add_conditional_edges("agent", tools_condition)
    # After a tool finishes, send the data back to Gemini to speak
    workflow.add_edge("tools", "agent")
    
    # Add memory so it can handle follow-up questions
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)