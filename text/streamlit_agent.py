import os
import sys
import streamlit as st
from datetime import datetime

sys.path.append(r"e:\AI_Customer\text")
from langgraph_flow import (
    DSN, init_human_intervention_table, save_conversation,
    get_conversation_history, get_pending_interventions, update_intervention_status,
    get_completed_interventions
)
import psycopg2

st.set_page_config(
    page_title="客服工作台",
    page_icon="🎧",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "agent_name" not in st.session_state:
    st.session_state.agent_name = ""
if "selected_intervention" not in st.session_state:
    st.session_state.selected_intervention = None
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

def check_agent_login(agent_name: str, password: str) -> bool:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM agents WHERE agent_name = %s AND password = %s', (agent_name, password))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return result is not None
    except Exception as e:
        st.error(f"登录验证失败: {e}")
        return False

def register_agent(agent_name: str, password: str) -> bool:
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO agents (agent_name, password) VALUES (%s, %s)', (agent_name, password))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"注册失败: {e}")
        return False

def save_agent_reply(intervention_id: int, agent_name: str, reply_content: str):
    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO agent_replies (intervention_id, agent_name, reply_content)
            VALUES (%s, %s, %s);
        """, (intervention_id, agent_name, reply_content))
        conn.commit()
    except Exception as e:
        print(f"Failed to save agent reply: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def load_conversation_messages(conversation_id: str) -> list:
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
        return messages
    except Exception as e:
        st.error(f"加载对话消息失败: {e}")
        return []

def login_page():
    st.title("🎧 客服工作台")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("客服登录")
        tab1, tab2 = st.tabs(["登录", "注册"])
        with tab1:
            with st.form("agent_login_form"):
                agent_name = st.text_input("客服账号", placeholder="请输入客服账号")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                submitted = st.form_submit_button("登录", use_container_width=True)
                if submitted:
                    if agent_name and password:
                        if check_agent_login(agent_name, password):
                            st.session_state.logged_in = True
                            st.session_state.agent_name = agent_name
                            init_human_intervention_table()
                            st.rerun()
                        else:
                            st.error("客服账号或密码错误")
                    else:
                        st.warning("请输入客服账号和密码")
        with tab2:
            with st.form("agent_register_form"):
                new_agent_name = st.text_input("客服账号", placeholder="请输入客服账号", key="reg_agent_name")
                new_password = st.text_input("密码", type="password", placeholder="请输入密码", key="reg_agent_password")
                confirm_password = st.text_input("确认密码", type="password", placeholder="请确认密码", key="reg_agent_confirm")
                submitted = st.form_submit_button("注册", use_container_width=True)
                if submitted:
                    if new_agent_name and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("两次输入的密码不一致")
                        elif register_agent(new_agent_name, new_password):
                            st.success("注册成功，请登录")
                        else:
                            st.error("客服账号已存在或注册失败")
                    else:
                        st.warning("请填写所有字段")

def agent_dashboard():
    st.title("🎧 客服工作台")
    st.markdown(f"👤 当前客服: **{st.session_state.agent_name}**")
    st.markdown("---")
    if st.session_state.selected_intervention:
        intervention_id = st.session_state.selected_intervention['id']
        interventions = get_pending_interventions()
        active_ids = [inv['id'] for inv in interventions]
        if intervention_id not in active_ids:
            st.session_state.selected_intervention = None
            st.session_state.agent_messages = []
    col1, col2 = st.columns([1, 2])
    with col1:
        st.header("工单列表")
        import time
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"⏰ 最后刷新: {current_time}")
        @st.fragment(run_every=5)
        def refresh_interventions():
            pending = get_pending_interventions()
            completed = get_completed_interventions()
            tab1, tab2 = st.tabs(["📥 未处理", "📤 已处理"])
            with tab1:
                if pending:
                    st.success(f"✅ {len(pending)} 条待处理工单")
                    for intervention in pending:
                        urgency_label = "🔴 加急" if intervention.get("urgency") == "urgent" else ""
                        expander_label = f"📋 {intervention['username']} - {intervention['created_at'].strftime('%m-%d %H:%M')} {urgency_label}"
                        with st.expander(expander_label):
                            if intervention.get("urgency") == "urgent":
                                st.error("🔴 急需处理")
                            st.write(f"**用户输入**: {intervention['user_input']}")
                            st.write(f"**情感**: {intervention['sentiment']}")
                            st.write(f"**意图**: {intervention['intent']}")
                            if intervention['order_info']:
                                st.write(f"**订单信息**: {intervention['order_info']}")
                            st.write(f"**工单ID**: {intervention['id']}")
                            st.write(f"**对话ID**: {intervention['conversation_id']}")
                            st.write(f"**状态**: {intervention['status']}")
                            if st.button("📞 接入", key=f"handle_{intervention['id']}", use_container_width=True):
                                st.session_state.selected_intervention = intervention
                                st.session_state.agent_messages = []
                                st.rerun()
                else:
                    st.info("暂无待处理的客服介入")
            with tab2:
                if completed:
                    st.success(f"✅ {len(completed)} 条已处理工单")
                    selected_status = st.selectbox("筛选状态", ["全部", "已完成", "已取消"], key="completed_filter")
                    filtered = completed
                    if selected_status == "已完成":
                        filtered = [c for c in completed if c['status'] == 'completed']
                    elif selected_status == "已取消":
                        filtered = [c for c in completed if c['status'] == 'cancelled']
                    for intervention in filtered:
                        status_icon = "✅" if intervention['status'] == 'completed' else "❌"
                        with st.expander(f"{status_icon} {intervention['username']} - {intervention['updated_at'].strftime('%m-%d %H:%M')}"):
                            st.write(f"**用户输入**: {intervention['user_input']}")
                            st.write(f"**情感**: {intervention['sentiment']}")
                            st.write(f"**意图**: {intervention['intent']}")
                            st.write(f"**最终状态**: {intervention['status']}")
                else:
                    st.info("暂无已处理的客服介入")
        refresh_interventions()
    with col2:
        if st.session_state.selected_intervention:
            intervention = st.session_state.selected_intervention
            st.header("对话详情")
            with st.expander("📊 上下文信息", expanded=True):
                st.markdown(f"**用户名**: {intervention['username']}")
                st.markdown(f"**对话ID**: {intervention['conversation_id']}")
                st.markdown(f"**用户输入**: {intervention['user_input']}")
                st.markdown(f"**情感分析**: {intervention['sentiment']}")
                st.markdown(f"**意图识别**: {intervention['intent']}")
                if intervention['order_info']:
                    st.markdown("**订单信息**:")
                    st.json(intervention['order_info'])
                else:
                    st.markdown("**订单信息**: 无")
            @st.fragment(run_every=3)
            def display_conversation():
                messages = load_conversation_messages(intervention['conversation_id'])
                st.session_state.agent_messages = messages
                st.subheader("对话历史")
                for msg in messages:
                    if msg["role"] == "user":
                        st.chat_message("user").markdown(f"**👤 用户**: {msg['content']}")
                        st.caption(f"情感: {msg.get('sentiment', '未知')} | 意图: {msg.get('intent', '未知')}")
                    else:
                        st.chat_message("assistant").markdown(f"**🤖 客服**: {msg['content']}")
            display_conversation()
            st.markdown("---")
            st.subheader("客服回复")
            reply_input = st.text_area("请输入回复内容", height=100, key="agent_reply", placeholder="请输入人工客服的回复内容...")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📤 发送回复", use_container_width=True, type="primary"):
                    if reply_input:
                        save_agent_reply(intervention['id'], st.session_state.agent_name, reply_input)
                        save_conversation(intervention['conversation_id'], "assistant", reply_input, intervention['sentiment'], intervention['intent'])
                        update_intervention_status(intervention['id'], 'processing')
                        st.success("回复已发送，等待用户下一条消息...")
                        st.session_state.agent_messages = []
                        st.rerun()
                    else:
                        st.warning("请输入回复内容")
            with col2:
                if st.button("❌ 结束对话", use_container_width=True):
                    update_intervention_status(intervention['id'], 'completed')
                    st.session_state.selected_intervention = None
                    st.session_state.agent_messages = []
                    st.rerun()
        else:
            st.info("请从左侧选择一个待处理的客服介入")

def main():
    if st.session_state.logged_in:
        agent_dashboard()
    else:
        login_page()

if __name__ == "__main__":
    main()
