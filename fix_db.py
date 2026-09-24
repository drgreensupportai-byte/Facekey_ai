import sqlite3

# Path to your database file shown in the explorer
db_path = "database/facekey.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0 NOT NULL;")
    conn.commit()
    print("Column 'is_admin' successfully added!")
except Exception as e:
    print(f"Error: {e}")
finally:
    conn.close()