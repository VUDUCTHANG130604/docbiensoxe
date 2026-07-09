import sqlite3
import datetime

DB_NAME = "biensoxe.db"

def create_table():
    """Tạo bảng nếu chưa có"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS plates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_plate_number(plate_text):
    """Lưu biển số vào DB"""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("INSERT INTO plates (plate_number) VALUES (?)", (plate_text,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"DB Error: {e}")
        return False

def get_all_plates():
    """Lấy danh sách biển số (cho mục đích debug/hiển thị list)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM plates ORDER BY timestamp DESC")
    data = c.fetchall()
    conn.close()
    return data