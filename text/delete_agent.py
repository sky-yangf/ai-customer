import psycopg2

DSN = "postgresql://yang:***@localhost:5432/postgres?sslmode=disable"

try:
    conn = psycopg2.connect(DSN)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM agents WHERE agent_name = 'kefu01'")
    conn.commit()
    cursor.execute("SELECT * FROM agents WHERE agent_name = 'kefu01'")
    result = cursor.fetchone()
    if result:
        print("Delete failed")
    else:
        print("Agent deleted. Re-register via streamlit_agent.py")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Failed: {e}")
