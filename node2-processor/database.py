import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='events.db'):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                sound_class TEXT,
                confidence REAL,
                decibel_level REAL,
                audio_path TEXT,
                probabilities TEXT,
                session_id TEXT
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    
    def insert_event(self, timestamp, sound_class, confidence, decibel_level, audio_path, probabilities, session_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (timestamp, sound_class, confidence, decibel_level, audio_path, probabilities, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, sound_class, confidence, decibel_level, audio_path, str(probabilities), session_id))
        event_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()
        return event_id
    
    def get_events(self, sound_class=None, limit=100, offset=0):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        if sound_class:
            cursor.execute('''
                SELECT id, timestamp, sound_class, confidence, decibel_level, audio_path
                FROM events
                WHERE sound_class = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (sound_class, limit, offset))
        else:
            cursor.execute('''
                SELECT id, timestamp, sound_class, confidence, decibel_level, audio_path
                FROM events
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [{
            'id': r[0],
            'timestamp': r[1],
            'sound_class': r[2],
            'confidence': r[3],
            'decibel_level': r[4],
            'audio_path': r[5]
        } for r in results]
