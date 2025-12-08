import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Handle all database operations for event storage and retrieval.
    """
    
    def __init__(self):
        self.conn_params = {
            'host': DB_HOST,
            'port': DB_PORT,
            'database': DB_NAME,
            'user': DB_USER,
            'password': DB_PASSWORD
        }
        self._init_tables()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.conn_params)
            return conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def _init_tables(self):
        """Create tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                sound_class VARCHAR(50) NOT NULL,
                confidence FLOAT NOT NULL,
                decibel_level FLOAT,
                audio_path VARCHAR(255),
                probabilities JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR(36)
            );
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_class ON events(sound_class);
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("Database tables initialized")
    
    def insert_event(self, timestamp, sound_class, confidence,
                     decibel_level, audio_path, probabilities, session_id):
        """
        Insert a detected sound event into database.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO events 
                (timestamp, sound_class, confidence, decibel_level, audio_path, probabilities, session_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
            """, (timestamp, sound_class, confidence, decibel_level, audio_path,
                  str(probabilities), session_id))
            
            event_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Stored event {event_id}: {sound_class} ({confidence:.2f})")
            return event_id
        
        except psycopg2.Error as e:
            logger.error(f"Error inserting event: {e}")
            conn.rollback()
            return None
        finally:
            cursor.close()
            conn.close()
    
    def get_events(self, sound_class=None, limit=100, offset=0):
        """
        Retrieve events with optional filtering.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if sound_class:
            cursor.execute("""
                SELECT id, timestamp, sound_class, confidence, decibel_level, audio_path
                FROM events
                WHERE sound_class = %s
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s;
            """, (sound_class, limit, offset))
        else:
            cursor.execute("""
                SELECT id, timestamp, sound_class, confidence, decibel_level, audio_path
                FROM events
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s;
            """, (limit, offset))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [
            {
                'id': r[0],
                'timestamp': r[1].isoformat(),
                'sound_class': r[2],
                'confidence': r[3],
                'decibel_level': r[4],
                'audio_path': r[5]
            }
            for r in results
        ]
