import sqlite3
from datetime import datetime

conn = sqlite3.connect("history.db", check_same_thread=False)

c = conn.cursor()

def init_db():
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    
    conn.commit()

def save_history(question, answer):
    timestamp = datetime.now().isoformat()
    c.execute(
    "INSERT INTO history (question, answer, timestamp) VALUES (?, ?, ?)",
    (question, answer, timestamp))

    conn.commit()

def get_history():
    c.execute("SELECT * FROM history")
    rows = c.fetchall()
    return rows