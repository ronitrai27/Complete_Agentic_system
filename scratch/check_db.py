import sqlite3

def check_db():
    conn = sqlite3.connect('data/conversations.db')
    cursor = conn.cursor()
    
    # Check if messages table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Tables:", tables)
    
    if ("messages",) in tables:
        cursor.execute("SELECT content FROM messages WHERE content LIKE '%Project%'")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} messages containing 'Project'")
        for r in rows[:10]:
            print("---")
            print(r[0][:300])

if __name__ == "__main__":
    check_db()
