import time
import sqlite3
import logging
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    db_result: str
    error: str
    retry_count: int
    intent: str 
    router_remarks: str

def build_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    schema = """Table: hospitals. Columns: "HOSPITAL NAME" (TEXT), "Address" (TEXT), "CITY" (TEXT)."""

    # --- NODE 1: INTENT CLASSIFIER ---
    def classify_intent_node(state: State):
        logger.info("\n[AGENT] 🧭 Classifying Intent...")
        
        prompt = """You are Loop AI's intent router. Analyze the user's request.
        1. GREETING: Reply with 'GREETING: <response>'
        2. REJECT: Reply with 'REJECT: <response>'
        3. SEARCH: Reply with 'SEARCH | <instructions for the SQL agent>'
        
        CRITICAL SEARCH EXAMPLES:
        - General suggestions: "SEARCH | Limit to 5 results."
        - Specific hospital: "SEARCH | User wants Yashwant Hospital. Force LIKE '%Yashwant%'."
        - Count request: "SEARCH | User is asking for a count. Use SELECT COUNT(*)."
        """
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages).content.strip()

        if response.startswith("GREETING:"):
            return {"intent": "direct", "db_result": response.replace("GREETING:", "").strip()}
        elif response.startswith("REJECT:"):
            return {"intent": "direct", "db_result": response.replace("REJECT:", "").strip()}
        elif response.startswith("SEARCH"):
            if "|" in response:
                remarks = response.split("|", 1)[1].strip()
            else:
                remarks = "Limit to 5 results."
            return {"intent": "search", "router_remarks": remarks}
        else:
            logger.warning(f"\n[SYSTEM] ⚠️ Unexpected router output. Defaulting to search. Output: {response}")
            return {"intent": "search", "router_remarks": "Limit to 5 results."}

    # --- NODE 2: DIRECT RESPONSE ---
    def direct_response_node(state: State):
        logger.info("\n[AGENT] 🗣️ Replying directly (Greeting/Out of Scope)")
        return {"messages": [AIMessage(content=state["db_result"])]}

    # --- NODE 3: SQL GENERATOR ---
    def generate_sql_node(state: State):
        remarks = state.get("router_remarks", "LIMIT query to 5 results.")
        logger.info(f"\n[AGENT] 🤖 SQL Generator thinking with remarks: {remarks}")
        
        prompt = f"""You are a SQLite expert. Output ONLY a raw SQL SELECT query based on this schema: {schema}

        >>> MANAGER INSTRUCTIONS FOR THIS TURN: <<<
        {remarks}
        
        >>> STRICT LOGIC RULES: <<<
        1. FUZZY VS EXACT: Default to `LIKE '%keyword%'`. Exception: If user says "only" (e.g., "Delhi only"), use `=`.
        2. CITY ALIASES: For "Bangalore", check `(CITY LIKE '%Bangalore%' OR CITY LIKE '%Bengaluru%')`.
        3. PAGINATION: If asking for "more", use `OFFSET` (e.g., `LIMIT 5 OFFSET 5`).
        4. CLARIFICATION: If the request is too vague, return exactly: 'CLARIFY'.
        """

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        
        sql = response.content.replace("```sql", "").replace("```", "").replace("ite", "").strip()
        
        if sql == "CLARIFY":
            return {"sql_query": "CLARIFY", "error": None, "retry_count": 0}
            
        if "SELECT" in sql.upper():
            sql = sql[sql.upper().find("SELECT"):]
            
        return {"sql_query": sql, "error": None, "retry_count": 0}

    # --- NODE 4: SQL EXECUTOR (With Observability) ---
    def execute_sql_node(state: State):
        query = state["sql_query"]
        logger.info(f"\n[SYSTEM] ⚙️ Executing SQL: {query}")
        
        if not query.upper().startswith("SELECT"):
            logger.error("\n[SYSTEM] 🛑 SECURITY BLOCK: Non-SELECT query detected.")
            return {"error": "Only SELECT queries are allowed.", "db_result": None}
            
        try:
            start_time = time.time()
            conn = sqlite3.connect('file:data/hospitals.db?mode=ro', uri=True, timeout=5.0)
            cursor = conn.cursor()
            conn.execute("PRAGMA busy_timeout = 3000")
            
            cursor.execute(query)
            results = cursor.fetchall()
            duration = time.time() - start_time
            
            logger.info(f"\n[SYSTEM] 📊 Rows returned: {len(results)} in {duration:.3f}s")
            
            if results:
                columns = [desc[0] for desc in cursor.description]
                formatted = [dict(zip(columns, row)) for row in results]
                result_str = str(formatted)
            else:
                result_str = "No results found."
                
            conn.close()
            return {"db_result": result_str, "error": None}
            
        except Exception as e:
            return {"error": str(e), "db_result": None}

    # --- NODE 5: ERROR REPAIR ---
    def repair_sql_node(state: State):
        logger.info(f"\n[AGENT] 🔧 Repair Agent fixing SQL (Attempt {state['retry_count'] + 1})...")
        prompt = f"""You are a SQLite repair expert. Your previous query failed.
        Schema: {schema}
        Bad Query: {state['sql_query']}
        Error Message: {state['error']}
        Return ONLY the raw valid SQLite SELECT query. No markdown."""
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        sql = response.content.replace("```sql", "").replace("```", "").strip()
        return {"sql_query": sql, "error": None, "retry_count": state["retry_count"] + 1}

    # --- NODE 6: SYNTHESIZER (State Isolated) ---
    def synthesize_node(state: State):
        # Cleanly handle clarification request
        if state.get("sql_query") == "CLARIFY":
            prompt = "You are 'Loop AI'. The user's request was too vague. Politely ask them to clarify what hospital or location they are looking for."
        else:
            db_data = state.get('db_result', 'No results found.')
            prompt = f"""You are 'Loop AI'. You must answer the user naturally using this specific data: {db_data}
            
            CRITICAL INTERPRETATION RULES:
            1. If the data is a single number in a list like '[(124,)]', that is the TOTAL count. Tell them the count.
            2. If the data is 'No results found', inform them politely. Do not make up data.
            """
        
        # State Isolation: Only send the prompt and the single latest message
        messages = [SystemMessage(content=prompt), state["messages"][-1]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # --- NODE 7: FAILURE HANDLING ---
    def synthesize_failure_node(state: State):
        msg = "I'm having trouble accessing the database right now. Please try again."
        return {"messages": [AIMessage(content=msg)]}

    # --- ROUTING LOGIC ---
    def route_intent(state: State):
        if state["intent"] == "search":
            return "generate_sql"
        return "direct_response"

    def route_sql_generation(state: State):
        if state.get("sql_query") == "CLARIFY":
            return "synthesize"
        return "execute_sql"

    def route_execution(state: State):
        if state.get("error"):
            if state.get("retry_count", 0) >= 2:
                logger.error("\n[SYSTEM] 🛑 Max retries reached.")
                return "synthesize_failure"
            return "repair_sql"
        return "synthesize"
    
    # --- BUILD THE GRAPH ---
    workflow = StateGraph(State)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("repair_sql", repair_sql_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("synthesize_failure", synthesize_failure_node)
    
    workflow.add_edge(START, "classify_intent")
    workflow.add_conditional_edges("classify_intent", route_intent)
    workflow.add_edge("direct_response", END)
    
    workflow.add_conditional_edges("generate_sql", route_sql_generation)
    workflow.add_conditional_edges("execute_sql", route_execution)
    workflow.add_edge("repair_sql", "execute_sql")
    
    workflow.add_edge("synthesize", END)
    workflow.add_edge("synthesize_failure", END)
    
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)