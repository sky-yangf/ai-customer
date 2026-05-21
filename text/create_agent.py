import psycopg2

DSN = "postgresql://yang:***@localhost:5432/postgres?sslmode=disable"

try:
    conn = psycopg2.connect(DSN)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id SERIAL PRIMARY KEY,
            agent_name VARCHAR(64) UNIQUE NOT NULL,
            password VARCHAR(128) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("DELETE FROM agents WHERE agent_name = 'kefu01'")
    cursor.execute("INSERT INTO agents (agent_name, password) VALUES ('kefu01', '123456')")
    conn.commit()
    cursor.execute("SELECT * FROM agents WHERE agent_name = 'kefu01'")
    result = cursor.fetchone()
    if result:
        print(f"Agent created: {result[1]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Failed: {e}")
