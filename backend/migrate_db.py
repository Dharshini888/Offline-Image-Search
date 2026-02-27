import sqlite3
import os

db_path = '../data/db.sqlite'
if os.path.exists(db_path):
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE images ADD COLUMN person_count INTEGER DEFAULT 0;")
        conn.commit()
        conn.close()
        print("Migration successful: added person_count column.")
    except Exception as e:
        print(f"Migration error: {e}")
else:
    print("Database not found.")
