import sqlite3
import logging, re
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
        logger.info("\n\n\n[AGENT] 🧭 Classifying Intent...")
        
        prompt = """You are Loop AI's intent router. Analyze the user's request.
        1. GREETING: If the user says hello or asks who you are, reply EXACTLY with: 'GREETING: Hello! I am Loop AI, your hospital network voice assistant. How can I help you find a hospital today?'
        2. HANDOFF: Reply with 'HANDOFF: Please hold while I transfer you.'
        3. REJECT: If completely out of scope, reply EXACTLY with: 'REJECT: I'm sorry, I can't help with that. I am forwarding this to a human agent.'
        4. UNCLEAR: If the text is gibberish, reply with 'UNCLEAR: I didn't quite catch that. Could you repeat?'
        5. PAGINATION: If the user asks for "more" or "next" hospitals, reply exactly with 'PAGINATION'
        6. FOLLOW_UP: If the user asks for a detail about a specific item from the previous response... reply with 'SEARCH | Rerun the exact same hospital search...'
        7. SEARCH: Reply with 'SEARCH | <semantic goal>'. 
           CRITICAL: DO NOT write SQL syntax here. Do not use `=` or `WHERE`.
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
        
        elif response_upper.startswith("UNCLEAR"):
            clean_res = response.split(":", 1)[1].strip() if ":" in response else response
            # This loops back to the user without hanging up
            return {"intent": "direct", "db_result": clean_res}
            
        elif response_upper.startswith("REJECT"):
            # This will now perfectly extract the exact assignment phrase
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

    # --- NODE 3: SQL GENERATOR ---
    def generate_sql_node(state: State):
        remarks = state.get("router_remarks", "")
        
        # --- DETERMINISTIC STATE TRACKING ---
        current_limit = state.get("last_limit", 5) 
        last_sql = state.get("last_sql", "")
        
        # Defensive string matching
        is_pagination = "PAGINATION" in remarks.strip().upper()
        
        if is_pagination and last_sql:
            current_offset = state.get("last_offset", 0) + current_limit
            injected_instruction = f"""
            User wants the NEXT page of results. 
            Modify this exact previous query:
            {last_sql}
            
            STRICTLY apply LIMIT {current_limit} OFFSET {current_offset}.
            Do NOT change the base WHERE clauses.
            """
        else:
            current_offset = 0
            # FIX 1: Tell the LLM to use our default 5 ONLY IF the user didn't specify a number
            injected_instruction = f"Manager Instructions: {remarks}. Apply LIMIT {current_limit} UNLESS the manager instructions specify a different limit."
            
        logger.info(f"\n[AGENT] 🤖 SQL Generator thinking: {injected_instruction}")
        
        prompt = f"""You are a SQLite expert. Output ONLY a raw SQL SELECT query. Schema: {schema}
        
        >>> INSTRUCTIONS FOR THIS TURN: <<<
        {injected_instruction}
        
        >>> STRICT LOGIC RULES: <<<
        1. DATA PREFETCHING: ALWAYS use `SELECT *` instead of selecting specific columns, UNLESS the manager instructions specifically ask for a count or total number. If asking for a count, you MUST use `SELECT COUNT(*)`.
        2. AVOID EXACT LONG STRINGS: NEVER filter `WHERE Address LIKE...` using a long, exact address string. It will cause the database to crash. Just filter by Hospital Name or City.
        3. EXTREME FUZZY MATCHING: You MUST ALWAYS use `LIKE '%keyword%'`. NEVER use `=`. Replace spaces with `%` (e.g., `"HOSPITAL NAME" LIKE '%Abhay%Hospital%'`).
        4. CITY ALIASES: For "Bangalore", check `(CITY LIKE '%Bangalore%' OR CITY LIKE '%Bengaluru%')`.
        5. CLARIFY: If request is too vague, return 'CLARIFY'.
        """

        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        sql = response.content.replace("```sql", "").replace("```", "").replace("ite", "").strip()
        
        if sql == "CLARIFY":
            return {"sql_query": "CLARIFY", "error": None, "retry_count": 0}
            
        if "SELECT" in sql.upper():
            sql = sql[sql.upper().find("SELECT"):]
            
        # --- PYTHON VERIFICATION (Trust, but Verify) ---
        if "LIMIT" not in sql.upper():
            logger.warning("\n[SYSTEM] ⚠️ LLM forgot LIMIT. Enforcing via Python.")
            if is_pagination:
                sql += f" LIMIT {current_limit} OFFSET {current_offset}"
            else:
                sql += f" LIMIT {current_limit}"
                
        # --- FIX 2: SYNCHRONIZE PYTHON STATE WITH LLM DECISION ---
        # If the LLM successfully generated 'LIMIT 12', we must update our Python state
        # so the next pagination jump is 12, not 5.
        limit_match = re.search(r'LIMIT\s+(\d+)', sql.upper())
        if limit_match:
            current_limit = int(limit_match.group(1))
                
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
    # --- NODE 6: SYNTHESIZER (The Human Voice) ---
    def synthesize_node(state: State):
        
        # FIX 1: Check if this is the very first turn of the conversation
        is_first_turn = len(state["messages"]) == 1
        intro_rule = ""
        if is_first_turn:
            intro_rule = "CRITICAL: This is the first interaction. You MUST start your response by saying 'Hello, I am Loop AI...'"
        
        if state.get("sql_query") == "CLARIFY":
            prompt = f"""You are 'Loop AI', a warm and helpful voice assistant. 
            {intro_rule}
            The user's request was a bit too vague to search. 
            Politely ask them to clarify which city or specific hospital name they are looking for."""
        
        else:
            db_data = state.get('db_result', 'No results found.')
            # --- NODE 6: SYNTHESIZER (The Human Voice) ---
    def synthesize_node(state: State):
        
        # FIX 1: Check if this is the very first turn of the conversation
        is_first_turn = len(state["messages"]) == 1
        intro_rule = ""
        if is_first_turn:
            intro_rule = "CRITICAL: This is the first interaction. You MUST start your response by saying 'Hello, I am Loop AI...'"
        
        if state.get("sql_query") == "CLARIFY":
            prompt = f"""You are 'Loop AI', a warm and helpful voice assistant. 
            {intro_rule}
            The user's request was a bit too vague to search. 
            Politely ask them to clarify which city or specific hospital name they are looking for."""
        
        else:
            db_data = state.get('db_result', 'No results found.')
            prompt = f"""You are 'Loop AI', a highly intelligent, empathetic voice assistant working for a hospital network.
            Your job is to translate raw database results into a natural, friendly spoken response.
            
            >>> RAW DATABASE RESULT: {db_data} <<<
            
            CRITICAL VOICE RULES:
            1. {intro_rule}
            2. SOUND HUMAN: Speak naturally using conversational transitions.
            3. NO JARGON: NEVER mention "SQL", "the database", or "tuples". 
            4. TTS FORMATTING (IMPORTANT): NEVER use markdown bullet points or asterisks (*). If listing multiple hospitals, use spoken numbers (e.g., "Number one...", "Number two...") or natural transition words (e.g., "First...", "Next...", "Finally...").
            5. EMPATHY ON EMPTY: If the data says 'No results found', say something like: "I'm so sorry, but I couldn't find any hospitals matching that right now."
            6. CONCISENESS (CRITICAL): The database result contains extra pre-fetched data (like full addresses). ONLY read out the specific details the user explicitly asked for. If they only asked for names, DO NOT read the addresses.
            """
        
        messages = [SystemMessage(content=prompt), state["messages"][-1]]
        response = llm.invoke(messages)
        return {"messages": [response]}
        
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