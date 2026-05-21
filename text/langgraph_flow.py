import os
import re
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import psycopg2

# ================= 全局配置 =================
MODEL_PATH = r"E:\AI_Customer\sentiment_ernie_v1\checkpoint-12657"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DSN = "postgresql://yang:***@localhost:5432/postgres?sslmode=disable"

INTENT_PRODUCT = "查询产品信息"
INTENT_ORDER = "查询订单"
INTENT_RETURN = "要求退货"
INTENT_HUMAN = "转人工客服"
INTENT_CHAT = "闲聊"

SENTIMENT_NEGATIVE = "消极"
SENTIMENT_NEUTRAL = "中性"
SENTIMENT_POSITIVE = "积极"

# ================= 预加载模型 =================
try:
    print(f"Loading sentiment model to {DEVICE}...")
    sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    sentiment_model.eval()
    print("Sentiment model loaded.")
except Exception as e:
    print(f"Failed to load sentiment model: {e}")

# ================= 状态定义 =================
class AgentState(TypedDict):
    user_input: str
    sentiment: str
    intent: str
    order_info: Optional[dict] = None
    product_info: Optional[str] = None
    llm_response: Optional[str] = None
    return_reason: Optional[str] = None
    need_human: bool = False
    sentiment_protected: bool = False
    final_response: Optional[str] = None
    history: list = None
    conversation_id: str = None
    thread_id: str = None

# ================= 数据库操作 =================
def init_conversation_table():
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                message_type VARCHAR(16) NOT NULL,
                content TEXT NOT NULL,
                sentiment VARCHAR(16),
                intent VARCHAR(32),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("conversation_history table initialized.")
    except Exception as e:
        print(f"Failed to init conversation_history: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def init_human_intervention_table():
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_intervention (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                username VARCHAR(64),
                user_input TEXT NOT NULL,
                sentiment VARCHAR(16),
                intent VARCHAR(32),
                order_info TEXT,
                status VARCHAR(16) DEFAULT 'pending',
                urgency VARCHAR(16) DEFAULT 'normal',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agent_replies (
                id SERIAL PRIMARY KEY,
                intervention_id INTEGER NOT NULL,
                agent_name VARCHAR(64) NOT NULL,
                reply_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (intervention_id) REFERENCES human_intervention(id)
            );
        """)
        conn.commit()
        print("human_intervention table initialized.")
    except Exception as e:
        print(f"Failed to init human_intervention: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def save_conversation(conversation_id: str, message_type: str, content: str,
                     sentiment: str = None, intent: str = None):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversation_history (conversation_id, message_type, content, sentiment, intent)
            VALUES (%s, %s, %s, %s, %s);
        """, (conversation_id, message_type, content, sentiment, intent))
        conn.commit()
    except Exception as e:
        print(f"Failed to save conversation: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_conversation_history(conversation_id: str, limit: int = 10) -> list:
    conn = None
    cursor = None
    history = []
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message_type, content, sentiment, intent, created_at
            FROM conversation_history
            WHERE conversation_id = %s
            ORDER BY created_at DESC
            LIMIT %s;
        """, (conversation_id, limit))
        results = cursor.fetchall()
        for row in reversed(results):
            history.append({
                "message_type": row[0], "content": row[1],
                "sentiment": row[2], "intent": row[3], "created_at": row[4]
            })
    except Exception as e:
        print(f"Failed to get conversation history: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return history

def save_human_intervention(conversation_id: str, username: str, user_input: str,
                           sentiment: str, intent: str, order_info: dict = None):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        import json
        order_info_json = json.dumps(order_info) if order_info else None
        cursor.execute("""
            SELECT id FROM human_intervention
            WHERE conversation_id = %s AND status IN ('pending', 'processing')
        """, (conversation_id,))
        existing = cursor.fetchone()
        if existing:
            return existing[0]
        urgency = evaluate_urgency(user_input, sentiment, intent)
        cursor.execute("""
            INSERT INTO human_intervention (conversation_id, username, user_input, sentiment, intent, order_info, urgency)
            VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id;
        """, (conversation_id, username, user_input, sentiment, intent, order_info_json, urgency))
        result = cursor.fetchone()
        conn.commit()
        return result[0] if result else None
    except Exception as e:
        print(f"Failed to save human intervention: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_pending_interventions() -> list:
    conn = None
    cursor = None
    interventions = []
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, conversation_id, username, user_input, sentiment, intent, order_info, status, created_at, urgency
            FROM human_intervention
            WHERE status IN ('pending', 'processing')
            ORDER BY CASE WHEN urgency = 'urgent' THEN 0 ELSE 1 END, created_at ASC;
        """)
        results = cursor.fetchall()
        for row in results:
            import json
            order_info = json.loads(row[6]) if row[6] else None
            interventions.append({
                "id": row[0], "conversation_id": row[1], "username": row[2],
                "user_input": row[3], "sentiment": row[4], "intent": row[5],
                "order_info": order_info, "status": row[7], "created_at": row[8],
                "urgency": row[9] if row[9] else "normal"
            })
    except Exception as e:
        print(f"Failed to get pending interventions: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return interventions

def get_completed_interventions(limit: int = 50) -> list:
    conn = None
    cursor = None
    interventions = []
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, conversation_id, username, user_input, sentiment, intent, order_info, status, created_at, updated_at, urgency
            FROM human_intervention
            WHERE status IN ('completed', 'cancelled')
            ORDER BY updated_at DESC
            LIMIT %s;
        """, (limit,))
        results = cursor.fetchall()
        for row in results:
            import json
            order_info = json.loads(row[6]) if row[6] else None
            interventions.append({
                "id": row[0], "conversation_id": row[1], "username": row[2],
                "user_input": row[3], "sentiment": row[4], "intent": row[5],
                "order_info": order_info, "status": row[7], "created_at": row[8],
                "updated_at": row[9], "urgency": row[10] if row[10] else "normal"
            })
    except Exception as e:
        print(f"Failed to get completed interventions: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return interventions

def evaluate_urgency(user_input: str, sentiment: str, intent: str) -> str:
    prompt = f"""判断工单紧急程度。用户输入：{user_input}，情感：{sentiment}，意图：{intent}。紧急输出"urgent"，否则输出"normal"。"""
    try:
        llm_response = call_dashscope_model(prompt)
        if llm_response and "urgent" in llm_response.lower():
            return "urgent"
        return "normal"
    except:
        return "normal"

def update_intervention_status(intervention_id: int, status: str):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE human_intervention SET status = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s;
        """, (status, intervention_id))
        conn.commit()
    except Exception as e:
        print(f"Failed to update intervention status: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def cancel_human_intervention(conversation_id: str) -> bool:
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE human_intervention SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
            WHERE conversation_id = %s AND status IN ('pending', 'processing') RETURNING id;
        """, (conversation_id,))
        result = cursor.fetchone()
        conn.commit()
        return True
    except Exception as e:
        print(f"Failed to cancel intervention: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_agent_reply_for_conversation(conversation_id: str) -> tuple:
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM human_intervention
            WHERE conversation_id = %s AND status IN ('pending', 'processing')
            ORDER BY created_at DESC LIMIT 1;
        """, (conversation_id,))
        intervention = cursor.fetchone()
        if intervention:
            cursor.execute("""
                SELECT reply_content, agent_name FROM agent_replies
                WHERE intervention_id = %s ORDER BY created_at DESC LIMIT 1;
            """, (intervention[0],))
            reply = cursor.fetchone()
            if reply:
                return (reply[0], reply[1])
    except Exception as e:
        print(f"Failed to get agent reply: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return None

def get_new_agent_replies(conversation_id: str, last_reply_id: int = None) -> list:
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM human_intervention
            WHERE conversation_id = %s AND status IN ('pending', 'processing')
            ORDER BY created_at DESC LIMIT 1;
        """, (conversation_id,))
        intervention = cursor.fetchone()
        if intervention:
            intervention_id = intervention[0]
            if last_reply_id:
                cursor.execute("""
                    SELECT id, reply_content, agent_name FROM agent_replies
                    WHERE intervention_id = %s AND id > %s
                    ORDER BY created_at ASC;
                """, (intervention_id, last_reply_id))
            else:
                cursor.execute("""
                    SELECT id, reply_content, agent_name FROM agent_replies
                    WHERE intervention_id = %s ORDER BY created_at ASC;
                """, (intervention_id,))
            replies = cursor.fetchall()
            return [{"id": r[0], "content": r[1], "agent_name": r[2]} for r in replies]
    except Exception as e:
        print(f"Failed to get new agent replies: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return []

def check_pending_agent_reply(conversation_id: str) -> bool:
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM human_intervention
            WHERE conversation_id = %s AND status = 'pending';
        """, (conversation_id,))
        return cursor.fetchone() is not None
    except Exception as e:
        print(f"Failed to check pending reply: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return False

def check_intervention_completed(conversation_id: str) -> bool:
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT status FROM human_intervention
            WHERE conversation_id = %s ORDER BY created_at DESC LIMIT 1;
        """, (conversation_id,))
        result = cursor.fetchone()
        return result is not None and result[0] == 'completed'
    except Exception as e:
        print(f"Failed to check intervention completed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return False

# ================= 辅助函数 =================
def build_context_prompt(history: list, current_input: str) -> str:
    context = ""
    if history:
        context = "对话历史：\n"
        for msg in history:
            role = "用户" if msg["message_type"] == "user" else "客服"
            context += f"{role}: {msg['content']}\n"
    return f"{context}\n当前用户输入：{current_input}"

# ================= LLM调用 =================
def call_dashscope_model(prompt: str) -> str:
    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to call dashscope: {e}")
        return None

# ================= 节点实现 =================
def sentiment_analysis_node(state: AgentState) -> AgentState:
    text = state["user_input"]
    if not text or not isinstance(text, str):
        state["sentiment"] = SENTIMENT_NEUTRAL
        return state
    clean_text = text.strip()
    inputs = sentiment_tokenizer(
        clean_text, padding=True, truncation=True, max_length=128, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    label_map = {0: SENTIMENT_NEGATIVE, 1: SENTIMENT_NEUTRAL, 2: SENTIMENT_POSITIVE}
    state["sentiment"] = label_map.get(predictions.item(), SENTIMENT_NEUTRAL)
    return state

def intent_recognition_node(state: AgentState) -> AgentState:
    query = state["user_input"]
    context_prompt = build_context_prompt(state.get("history", []), query)
    prompt = f"""
    判断用户意图：1=查询订单, 2=查询产品, 3=要求退货, 4=转人工, 5=闲聊。
    {context_prompt}
    只输出数字编号。
    """
    llm_response = call_dashscope_model(prompt)
    intent_map = {
        "1": INTENT_ORDER, "2": INTENT_PRODUCT, "3": INTENT_RETURN,
        "4": INTENT_HUMAN, "5": INTENT_CHAT
    }
    if llm_response:
        num_match = re.search(r'\d', llm_response)
        if num_match:
            state["intent"] = intent_map.get(num_match.group(), INTENT_CHAT)
        else:
            state["intent"] = INTENT_CHAT
    else:
        state["intent"] = INTENT_CHAT
    return state

def sentiment_protection_node(state: AgentState) -> AgentState:
    if state["sentiment"] == SENTIMENT_NEGATIVE:
        state["sentiment_protected"] = True
        context_prompt = build_context_prompt(state.get("history", []), state["user_input"])
        prompt = f"你是客服助手。{context_prompt} 用户情绪消极，请生成情感安抚回复：理解+道歉+愿意帮助。"
        llm_response = call_dashscope_model(prompt)
        state["llm_response"] = llm_response or "非常抱歉给您带来不好的体验，我们会尽力帮您解决。"
    return state

def order_query_node(state: AgentState) -> AgentState:
    user_input = state["user_input"]
    match = re.search(r'ORD\d+', user_input)
    order_id = match.group(0) if match else ''.join([c for c in user_input if c.isdigit()])
    if not order_id:
        state["order_info"] = None
        prompt = f"用户查询订单但未提供订单号：{user_input}。请礼貌请求提供订单号。"
        state["llm_response"] = call_dashscope_model(prompt) or "请提供您的订单号，我来帮您查询。"
        return state
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN, options='-c default_transaction_read_only=on')
        cursor = conn.cursor()
        cursor.execute('SELECT status, product_name FROM "order" WHERE order_no = %s', (order_id,))
        result = cursor.fetchone()
        if result:
            state["order_info"] = {"status": result[0], "product_name": result[1]}
        else:
            state["order_info"] = None
            state["llm_response"] = f"未找到订单号 '{order_id}' 的记录。"
    except Exception as e:
        print(f"Order query error: {e}")
        state["order_info"] = None
        state["llm_response"] = "查询订单时出现错误，请稍后重试。"
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return state

def product_query_node(state: AgentState) -> AgentState:
    state["product_info"] = "RAG产品检索功能正在开发中。"
    state["llm_response"] = "关于产品信息的查询功能正在完善中。"
    return state

def llm_chat_node(state: AgentState) -> AgentState:
    prompt = f"你是客服助手。用户说：\"{state['user_input']}\"。请生成友好的闲聊回复。"
    state["llm_response"] = call_dashscope_model(prompt) or "很高兴与您聊天！有什么可以帮助您的吗？"
    return state

def return_process_node(state: AgentState) -> AgentState:
    state["return_reason"] = "待用户回复"
    order_info = state.get("order_info")
    user_input = state["user_input"]
    new_order_match = re.search(r'ORD\d+', user_input)
    if new_order_match:
        conn = None
        cursor = None
        try:
            conn = psycopg2.connect(DSN, options='-c default_transaction_read_only=on')
            cursor = conn.cursor()
            cursor.execute('SELECT status, product_name FROM "order" WHERE order_no = %s', (new_order_match.group(0),))
            result = cursor.fetchone()
            if result:
                order_info = {"status": result[0], "product_name": result[1]}
                state["order_info"] = order_info
        except:
            pass
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    if order_info:
        prompt = f"用户要求退货。订单：{order_info}。请说明退货流程。"
    else:
        prompt = f"用户要求退货：{user_input}。请说明退货流程。"
    state["llm_response"] = call_dashscope_model(prompt) or "请问您是因为什么原因想要退货呢？"
    return state

def human_intervention_node(state: AgentState) -> AgentState:
    state["need_human"] = True
    state["final_response"] = "您好，正在为您转接人工客服，请稍候..."
    return state

def response_node(state: AgentState) -> AgentState:
    if state.get("final_response"):
        return state
    state["final_response"] = state.get("llm_response", "抱歉，未能获取有效回复。")
    return state

# ================= 路由函数 =================
def route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "")
    if intent == INTENT_ORDER:
        return "order_query"
    elif intent == INTENT_PRODUCT:
        return "product_query"
    elif intent == INTENT_RETURN:
        return "return_process"
    elif intent == INTENT_HUMAN:
        return "human_intervention"
    return "llm_chat"

def route_after_sentiment(state: AgentState) -> str:
    if state.get("sentiment") == SENTIMENT_NEGATIVE:
        return "sentiment_protection"
    return "intent_recognition"

def should_human_intervene(state: AgentState) -> bool:
    return state.get("need_human", False)

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("sentiment_analysis", sentiment_analysis_node)
    workflow.add_node("intent_recognition", intent_recognition_node)
    workflow.add_node("sentiment_protection", sentiment_protection_node)
    workflow.add_node("order_query", order_query_node)
    workflow.add_node("product_query", product_query_node)
    workflow.add_node("llm_chat", llm_chat_node)
    workflow.add_node("return_process", return_process_node)
    workflow.add_node("human_intervention", human_intervention_node)
    workflow.add_node("response", response_node)

    workflow.set_entry_point("sentiment_analysis")
    workflow.add_edge("sentiment_analysis", "intent_recognition")
    workflow.add_edge("intent_recognition", "order_query",
                      condition=lambda s: s.get("intent") == INTENT_ORDER)
    workflow.add_edge("intent_recognition", "product_query",
                      condition=lambda s: s.get("intent") == INTENT_PRODUCT)
    workflow.add_edge("intent_recognition", "return_process",
                      condition=lambda s: s.get("intent") == INTENT_RETURN)
    workflow.add_edge("intent_recognition", "human_intervention",
                      condition=lambda s: s.get("intent") == INTENT_HUMAN)
    workflow.add_edge("intent_recognition", "llm_chat",
                      condition=lambda s: s.get("intent") not in [INTENT_ORDER, INTENT_PRODUCT, INTENT_RETURN, INTENT_HUMAN])
    workflow.add_edge("order_query", "response")
    workflow.add_edge("product_query", "response")
    workflow.add_edge("llm_chat", "response")
    workflow.add_edge("return_process", "response")
    workflow.add_edge("sentiment_protection", "response")
    workflow.add_edge("human_intervention", "response")
    workflow.add_edge("response", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)
