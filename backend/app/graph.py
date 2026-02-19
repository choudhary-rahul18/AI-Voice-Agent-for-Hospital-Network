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

# --- 1. UPDATED STATE (Added retry_count and intent tracking) ---
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sql_query: str
    db_result: str
    error: str
    retry_count: int
    intent: str 
    router_remarks: str
    last_limit: int   
    last_offset: int  
    last_sql: str # Python tracks where we left off

def build_graph():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    schema = """Table: hospitals. Columns: "HOSPITAL NAME" (TEXT), "Address" (TEXT), "CITY" (TEXT)."""

    # --- NODE 1: INTENT CLASSIFIER (Fix 1) ---
    def classify_intent_node(state: State):
        logger.info("\n[AGENT] 🧭 Classifying Intent...")
        
        prompt = """You are Loop AI's intent router. Analyze the user's request.
        1. GREETING: Reply with 'GREETING: <response>'
        2. HANDOFF: Reply with 'HANDOFF: Please hold while I transfer you.'
        3. REJECT: Reply with 'REJECT: <response>'
        4. PAGINATION: If the user asks for "more" or "next" hospitals from the PREVIOUS search, reply exactly with 'PAGINATION'
        5. SEARCH: Reply with 'SEARCH | <instructions for the SQL agent>'
        """
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages).content.strip()
        response_upper = response.upper()

        if response_upper.startswith("GREETING"):
            clean_res = response.split(":", 1)[1].strip() if ":" in response else response
            return {"intent": "direct", "db_result": clean_res}
            
        elif response_upper.startswith("HANDOFF"):
            clean_res = response.split(":", 1)[1].strip() if ":" in response else response
            return {"intent": "handoff", "db_result": clean_res}
            
        elif response_upper.startswith("REJECT"):
            clean_res = response.split(":", 1)[1].strip() if ":" in response else response
            return {"intent": "direct", "db_result": clean_res}
            
        elif response_upper.startswith("PAGINATION"):
            return {"intent": "search", "router_remarks": "PAGINATION"}
            
        elif response_upper.startswith("SEARCH"):
            remarks = response.split("|", 1)[1].strip() if "|" in response else "Limit to 5 results."
            return {"intent": "search", "router_remarks": remarks}
            
        else:
            # THE SAFE FALLBACK: Do not assume search. Assume confusion.
            logger.warning(f"\n[SYSTEM] ⚠️ Unexpected router output: {response}")
            return {
                "intent": "direct", 
                "db_result": "I didn't quite catch that. Are you looking for hospital information?"
            }
        

    # --- NODE 2: DIRECT RESPONSE (Handles Greeting/Rejection) ---
    def direct_response_node(state: State):
        logger.info("\n[AGENT] 🗣️ Replying directly (Greeting/Out of Scope)")
        return {"messages": [AIMessage(content=state["db_result"])]}

    # --- NODE 3: SQL GENERATOR (Fix 5: Context retained) ---
    def generate_sql_node(state: State):
        remarks = state.get("router_remarks", "")
        
        # --- DETERMINISTIC STATE TRACKING ---
        current_limit = state.get("last_limit", 5) 
        last_sql = state.get("last_sql", "")
        
        # Defensive string matching (Fix 3)
        is_pagination = "PAGINATION" in remarks.strip().upper()
        
        if is_pagination and last_sql:
            current_offset = state.get("last_offset", 0) + current_limit
            # Fix 5: Inject exact previous query to prevent drift
            injected_instruction = f"""
            User wants the NEXT page of results. 
            Modify this exact previous query:
            {last_sql}
            
            STRICTLY apply LIMIT {current_limit} OFFSET {current_offset}.
            Do NOT change the base WHERE clauses.
            """
        else:
            current_offset = 0
            injected_instruction = f"Manager Instructions: {remarks}. Apply LIMIT {current_limit}."
            
        logger.info(f"\n[AGENT] 🤖 SQL Generator thinking: {injected_instruction}")
        
        prompt = f"""You are a SQLite expert. Output ONLY a raw SQL SELECT query. Schema: {schema}
        
        >>> INSTRUCTIONS FOR THIS TURN: <<<
        {injected_instruction}
        
        >>> STRICT LOGIC RULES: <<<
        1. FUZZY MATCHING: Default to `LIKE '%keyword%'`. 
        2. CITY ALIASES: For "Bangalore", check `(CITY LIKE '%Bangalore%' OR CITY LIKE '%Bengaluru%')`.
        3. CLARIFY: If request is too vague, return 'CLARIFY'.
        """

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        sql = response.content.replace("```sql", "").replace("```", "").replace("ite", "").strip()
        
        if sql == "CLARIFY":
            return {"sql_query": "CLARIFY", "error": None, "retry_count": 0}
            
        if "SELECT" in sql.upper():
            sql = sql[sql.upper().find("SELECT"):]
            
        # --- FIX 2: PYTHON VERIFICATION (Trust, but Verify) ---
        if "LIMIT" not in sql.upper():
            logger.warning("\n[SYSTEM] ⚠️ LLM forgot LIMIT. Enforcing via Python.")
            if is_pagination:
                sql += f" LIMIT {current_limit} OFFSET {current_offset}"
            else:
                sql += f" LIMIT {current_limit}"
                
        # Save state for the next potential pagination turn
        return {
            "sql_query": sql, 
            "error": None, 
            "retry_count": 0,
            "last_limit": current_limit,
            "last_offset": current_offset,
            "last_sql": sql
        }


    # --- NODE 4: SQL EXECUTOR (Fix 2, 6, 7: Security, Formatting, Timeouts) ---
    def execute_sql_node(state: State):
        query = state["sql_query"]
        logger.info(f"\n[SYSTEM] ⚙️ Executing SQL: {query}")
        
        # Fix 2: SQL Injection / Mutation Prevention
        if not query.upper().startswith("SELECT"):
            logger.error("\n[SYSTEM] 🛑 SECURITY BLOCK: Non-SELECT query detected.")
            return {"error": "Only SELECT queries are allowed.", "db_result": None}
            
        try:
            # Fix 7: Timeouts
            conn = sqlite3.connect('file:data/hospitals.db?mode=ro', uri=True, timeout=5.0)
            cursor = conn.cursor()
            
            # Additional PRAGMA safeguard
            conn.execute("PRAGMA busy_timeout = 3000")
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Fix 6: Dictionary formatting
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

    # --- NODE 5: ERROR REPAIR (Fix 3: Bounded Reflection) ---
    def repair_sql_node(state: State):
        logger.info(f"\n[AGENT] 🔧 Repair Agent fixing SQL (Attempt {state['retry_count'] + 1})...")
        prompt = f"""You are a SQLite repair expert. Your previous query failed.
        Schema: {schema}
        Bad Query: {state['sql_query']}
        Error Message: {state['error']}
        
        CRITICAL REPAIR RULES:
        1. FUZZY MATCHING: If you are fixing a WHERE clause, ALWAYS use `LIKE '%keyword%'` instead of `=`. 
           (Example: Change `CITY = 'Dwarka'` to `CITY LIKE '%Dwarka%'`).
        2. Return ONLY the raw valid SQLite SELECT query. Do not write explanations.
        """
        
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        sql = response.content.replace("```sql", "").replace("```", "").strip()
        
        return {"sql_query": sql, "error": None, "retry_count": state["retry_count"] + 1}

    # --- NODE 6 & 7: SYNTHESIZERS ---
    # --- NODE 6: SYNTHESIZER (The Human Voice) ---
    def synthesize_node(state: State):
        # 1. Handle Clarification with a conversational tone
        if state.get("sql_query") == "CLARIFY":
            prompt = """You are 'Loop AI', a warm and helpful voice assistant. 
            The user's request was a bit too vague to search. 
            Politely and naturally ask them to clarify which city or specific hospital name they are looking for."""
        
        # 2. Handle Database Results like a human
        else:
            db_data = state.get('db_result', 'No results found.')
            prompt = f"""You are 'Loop AI', a highly intelligent, empathetic, and conversational voice assistant working for a hospital network.
            Your job is to translate raw database results into a natural, friendly spoken response.
            
            >>> RAW DATABASE RESULT: {db_data} <<<
            
            CRITICAL VOICE RULES:
            1. SOUND HUMAN: Speak naturally. Use conversational transitions (e.g., "I found a few options for you," or "Sure thing!").
            2. NO JARGON: NEVER mention "SQL", "the database", "lists", or "tuples". 
            3. HANDLING COUNTS: If the data is a single number like '[(124,)]', say it naturally: "We currently have 124 hospitals in that area."
            4. HANDLING LISTS: If giving a list of addresses or names, read them clearly. 
            5. EMPATHY ON EMPTY: If the data says 'No results found', be polite and helpful. Say something like: "I'm so sorry, but I couldn't find any hospitals matching that right now. Could we try a different location?"
            """
        
        # State Isolation: Only send the prompt and the user's latest message to prevent confusion
        messages = [SystemMessage(content=prompt), state["messages"][-1]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def synthesize_failure_node(state: State):
        msg = "I'm having trouble accessing the database right now. Please try again."
        return {"messages": [AIMessage(content=msg)]}

    # --- ROUTING LOGIC ---
    def route_intent(state: State):
        if state["intent"] == "search":
            return "generate_sql"
        if state["intent"] in ["direct", "handoff"]:
            return "direct_response"
        return "direct_response" # Ultimate safety catch


    def route_execution(state: State):
        if state["sql_query"] == "CLARIFY":
            return "synthesize" # Let the synthesizer ask the user for details
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
    
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_conditional_edges("execute_sql", route_execution)
    
    workflow.add_edge("repair_sql", "execute_sql")
    
    workflow.add_edge("synthesize", END)
    workflow.add_edge("synthesize_failure", END)
    
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)