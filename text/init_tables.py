import psycopg2

DSN = "postgresql://yang:***@localhost:5432/postgres?sslmode=disable"

def init_all_tables():
    try:
        conn = psycopg2.connect(DSN)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS human_intervention (
                id SERIAL PRIMARY KEY,
                conversation_id VARCHAR(64) NOT NULL,
                username VARCHAR(64) NOT NULL,
                user_input TEXT NOT NULL,
                sentiment VARCHAR(16),
                intent VARCHAR(32),
                order_info TEXT,
                status VARCHAR(16) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intervention_status ON human_intervention(status);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_intervention_conversation ON human_intervention(conversation_id);")
        conn.commit()
        print("Tables initialized.")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    init_all_tables()
