import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('magbot.db')
c = conn.cursor()

# Create a table for users
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
''')

# Create a table for leave requests
c.execute('''
CREATE TABLE IF NOT EXISTS leave_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    leave_type TEXT,
    leave_duration INTEGER,
    reason TEXT,
    status TEXT DEFAULT 'Pending',
    FOREIGN KEY (user_id) REFERENCES users (id)
)
''')

conn.commit()
conn.close()

print("Database and tables initialized successfully.")
