import os
import sys
import streamlit as st
import shortuuid
from datetime import datetime

sys.path.append(r"e:\AI_Customer\text")
from langgraph_flow import (
    build_graph, call_dashscope_model, DSN, init_conversation_table,
    save_conversation, get_conversation_history, AgentState,
    save_human_intervention, get_new_agent_replies, check_intervention_completed,
    cancel_human_intervention
)
import psycopg2

st.set_page_config(
    page_title="AI客服系统",
    page_icon="🤖",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "app" not in st.session_state:
    st.session_state.app = None
if "waiting_for_human" not in st.session_state:
    st.session_state.waiting_for_human = False
if "last_agent_reply_id" not in st.session_state:
    st.session_state.last_agent_reply_id = None

def init_db():
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(64) UNIQUE NOT NULL,
                password VARCHAR(128) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"数据库初始化失败: {e}")
        return False

def check_login(username: str, password: str) -> bool:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = %s AND password = %s', (username, password))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Exception as e:
        st.error(f"登录验证失败: {e}")
        return False

def register_user(username: str, password: str) -> bool:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (%s, %s)', (username, password))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"注册失败: {e}")
        return False

def get_user_history(username: str) -> list:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT DISTINCT conversation_id, created_at
            FROM conversation_history
            WHERE conversation_id LIKE %s
            ORDER BY created_at DESC
            LIMIT 10
        """, (f"{username}-%",))
        result = cursor.fetchall()
        cursor.close()
        conn.close()
        return result
    except Exception as e:
        return []

def load_history(conversation_id: str) -> None:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT message_type, content, sentiment, intent, created_at
            FROM conversation_history
            WHERE conversation_id = %s
            ORDER BY created_at ASC
        """, (conversation_id,))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        messages = []
        for row in results:
            role = "user" if row[0] == "user" else "assistant"
            messages.append({
                "role": role, "content": row[1],
                "sentiment": row[2], "intent": row[3], "created_at": row[4]
            })
        st.session_state.messages = messages
    except Exception as e:
        st.error(f"加载历史对话失败: {e}")
        st.session_state.messages = []

def delete_conversation(conversation_id: str) -> bool:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversation_history WHERE conversation_id = %s", (conversation_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"删除对话失败: {e}")
        return False

def login_page():
    st.title("🤖 AI客服系统")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("用户登录")
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                submitted = st.form_submit_button("登录", use_container_width=True)
                if submitted:
                    if username and password:
                        if check_login(username, password):
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.conversation_id = f"{username}-{shortuuid.uuid()[:8]}"
                            init_conversation_table()
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")
                    else:
                        st.warning("请输入用户名和密码")
        with tab2:
            with st.form("register_form"):
                new_username = st.text_input("用户名", placeholder="请输入用户名", key="reg_username")
                new_password = st.text_input("密码", type="password", placeholder="请输入密码", key="reg_password")
                confirm_password = st.text_input("确认密码", type="password", placeholder="请确认密码", key="reg_confirm")
                submitted = st.form_submit_button("注册", use_container_width=True)
                if submitted:
                    if new_username and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("两次输入的密码不一致")
                        elif register_user(new_username, new_password):
                            st.success("注册成功，请登录")
                        else:
                            st.error("用户名已存在或注册失败")
                    else:
                        st.warning("请填写所有字段")

def chat_page():
    if not st.session_state.app:
        with st.spinner("正在初始化对话系统..."):
            st.session_state.app = build_graph()
    if not st.session_state.messages:
        load_history(st.session_state.conversation_id)
    st.title("🤖 AI客服助手")
    with st.sidebar:
        st.header("用户信息")
        st.write(f"👤 用户名: {st.session_state.username}")
        st.write(f"🆔 对话ID: {st.session_state.conversation_id}")
        st.markdown("---")
        if st.button("➕ 开始新对话", use_container_width=True, type="primary"):
            st.session_state.conversation_id = f"{st.session_state.username}-{shortuuid.uuid()[:8]}"
            st.session_state.messages = []
            st.rerun()
        st.markdown("---")
        st.header("历史对话")
        history = get_user_history(st.session_state.username)
        if history:
            st.markdown("选择一个对话继续：")
            for idx, (conv_id, created_at) in enumerate(history):
                button_key = f"history_{idx}_{conv_id[:8]}"
                delete_key = f"delete_{idx}_{conv_id[:8]}"
                is_active = conv_id == st.session_state.conversation_id
                button_label = f"📝 {conv_id.split('-')[1][:8]} ({created_at.strftime('%m-%d %H:%M')})"
                col1, col2 = st.columns([4, 1])
                with col1:
                    if is_active:
                        st.markdown(f"**✅ {button_label}**")
                    else:
                        if st.button(button_label, key=button_key, use_container_width=True):
                            st.session_state.conversation_id = conv_id
                            load_history(conv_id)
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=delete_key, help="删除此对话"):
                        if delete_conversation(conv_id):
                            st.success("对话已删除")
                            if conv_id == st.session_state.conversation_id:
                                st.session_state.messages = []
                            st.rerun()
        else:
            st.info("暂无历史对话")
        st.markdown("---")
        if st.button("退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.conversation_id = ""
            st.session_state.messages = []
            st.rerun()
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(f"**👤 您**: {msg['content']}")
                    if msg.get("sentiment"):
                        st.caption(f"情感: {msg['sentiment']} | 意图: {msg.get('intent', '未知')}")
            else:
                with st.chat_message("assistant"):
                    st.markdown(f"**🤖 客服**: {msg['content']}")
                    if msg.get("sentiment"):
                        st.caption(f"情感: {msg['sentiment']} | 意图: {msg.get('intent', '未知')}")
    if st.session_state.waiting_for_human:
        st.info("👨‍💼 人工客服会话中")
        @st.fragment(run_every=3)
        def human_chat_fragment():
            if check_intervention_completed(st.session_state.conversation_id):
                st.session_state.waiting_for_human = False
                st.session_state.last_agent_reply_id = None
                st.session_state.messages.append({
                    "role": "assistant", "content": "已结束人工服务，请问还有什么可以帮您？",
                    "sentiment": "中性", "intent": "闲聊"
                })
                save_conversation(st.session_state.conversation_id, "assistant", "已结束人工服务，请问还有什么可以帮您？", "中性", "闲聊")
                st.rerun()
                return
            new_replies = get_new_agent_replies(st.session_state.conversation_id, st.session_state.last_agent_reply_id)
            for reply in new_replies:
                st.session_state.messages.append({
                    "role": "assistant", "content": reply["content"],
                    "sentiment": "未知", "intent": "转人工客服"
                })
                st.session_state.last_agent_reply_id = reply["id"]
                st.success(f"✅ 人工客服 {reply['agent_name']} 回复了")
                st.rerun()
        human_chat_fragment()
        st.markdown("---")
        col1, col2 = st.columns([4, 1])
        with col1:
            if prompt := st.chat_input("继续输入...", key="human_chat_input"):
                user_input = prompt
                with st.chat_message("user"):
                    st.markdown(f"**👤 您**: {user_input}")
                save_conversation(st.session_state.conversation_id, "user", user_input, "未知", "转人工客服")
                st.session_state.messages.append({
                    "role": "user", "content": user_input, "sentiment": "未知", "intent": "转人工客服"
                })
                st.info("👨‍💼 等待人工客服回复...")
        with col2:
            if st.button("❌ 取消人工服务", key="cancel_human_service", use_container_width=True):
                if cancel_human_intervention(st.session_state.conversation_id):
                    st.session_state.waiting_for_human = False
                    st.session_state.last_agent_reply_id = None
                    st.session_state.messages.append({
                        "role": "assistant", "content": "已取消人工服务，请问还有什么可以帮您？",
                        "sentiment": "中性", "intent": "闲聊"
                    })
                    save_conversation(st.session_state.conversation_id, "assistant", "已取消人工服务，请问还有什么可以帮您？", "中性", "闲聊")
                    st.rerun()
                else:
                    st.error("取消失败，请重试")
    elif not st.session_state.waiting_for_human:
        if prompt := st.chat_input("请输入您的问题...", key="chat_input"):
            user_input = prompt
            with st.chat_message("user"):
                st.markdown(f"**👤 您**: {user_input}")
            with st.spinner("正在处理..."):
                try:
                    config = {"configurable": {"thread_id": st.session_state.conversation_id}}
                    result = st.session_state.app.invoke(
                        {"user_input": user_input, "history": [], "conversation_id": st.session_state.conversation_id},
                        config
                    )
                    state = st.session_state.app.get_state(config)
                    if state.next and len(state.next) and "human_intervention" in state.next:
                        current_state = state.values
                        sentiment = current_state.get("sentiment", "未知")
                        intent = current_state.get("intent", "未知")
                        order_info = current_state.get("order_info")
                        save_conversation(st.session_state.conversation_id, "user", user_input, sentiment, intent)
                        save_human_intervention(
                            conversation_id=st.session_state.conversation_id,
                            username=st.session_state.username,
                            user_input=user_input,
                            sentiment=sentiment, intent=intent, order_info=order_info
                        )
                        save_conversation(st.session_state.conversation_id, "assistant", "您好，正在为您转接人工客服，请稍候...", sentiment, intent)
                        st.session_state.messages.append({"role": "user", "content": user_input, "sentiment": sentiment, "intent": intent})
                        st.session_state.messages.append({"role": "assistant", "content": "您好，正在为您转接人工客服，请稍候...", "sentiment": sentiment, "intent": intent})
                        st.session_state.waiting_for_human = True
                        st.rerun()
                    else:
                        response = result.get("final_response", result.get("llm_response", "抱歉，我没有收到有效的回复。"))
                        sentiment = result.get("sentiment", "未知")
                        intent = result.get("intent", "未知")
                        save_conversation(st.session_state.conversation_id, "user", user_input, sentiment, intent)
                        save_conversation(st.session_state.conversation_id, "assistant", response, sentiment, intent)
                        st.session_state.messages.append({"role": "user", "content": user_input, "sentiment": sentiment, "intent": intent})
                        st.session_state.messages.append({"role": "assistant", "content": response, "sentiment": sentiment, "intent": intent})
                        st.rerun()
                except Exception as e:
                    st.error(f"处理失败: {e}")
    with st.expander("📊 当前状态信息"):
        if st.session_state.messages:
            last_msg = st.session_state.messages[-1]
            if last_msg["role"] == "assistant":
                st.json({
                    "用户输入": st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else "",
                    "情感分析": last_msg.get("sentiment", "未知"),
                    "意图识别": last_msg.get("intent", "未知"),
                    "对话ID": st.session_state.conversation_id
                })

def main():
    init_db()
    if st.session_state.logged_in:
        chat_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
