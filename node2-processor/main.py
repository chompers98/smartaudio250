from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import wave
import logging
from datetime import datetime
from config import *
from ml_inference import SoundClassifier
from database import DatabaseManager
import os
from pathlib import Path

#AUDIO_SAVE_DIR = Path("./audio_clips")
#AUDIO_SAVE_DIR.mkdir(exist_ok=True)

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

app = FastAPI(title="Sound Detection Node 2")

# load models and services
classifier = SoundClassifier()
db = DatabaseManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Node2 Processing"}

@app.post("/classify")
async def classify_sound(
    audio: UploadFile = File(...),
    timestamp: str = Form(...),
    decibel_level: float = Form(...),
    session_id: str = Form(...)
):
    """
    Main classification endpoint.
    Receives audio clip from Node 1 and returns classification.
    """
    try:
        # eead uploaded audio
        audio_bytes = await audio.read()
        
#        # DEBUGGING: save wav to disk
#        filename = f"audio_clip_{timestamp.replace(':', '-').replace('.', '_')}.wav"
#        filepath = AUDIO_SAVE_DIR / filename
#        with open(filepath, 'wb') as f:
#            f.write(audio_bytes)
#        logger.info(f"Saved audio to: {filepath}")
        
        # parse WAV file to get raw audio samples
        audio_data = parse_wav_data(audio_bytes)
        if audio_data is None:
            return JSONResponse(
                {"error": "Failed to parse audio"},
                status_code=400
            )
        
        logger.info(f"Raw audio shape: {audio_data.shape}")
                
        # run inference on raw audio
        prediction = classifier.predict(audio_data)
        
        # store in database
        db_result = db.insert_event(
            timestamp=datetime.fromisoformat(timestamp.replace('Z', '+00:00')),
            sound_class=prediction['class'],
            confidence=prediction['confidence'],
            decibel_level=decibel_level,
            audio_path=str(filepath),
            probabilities=prediction['probabilities'],
            session_id=session_id
        )
        
        return {
            "classification": prediction['class'],
            "confidence": prediction['confidence'],
            "decibel_level": decibel_level,
            "all_probabilities": prediction['probabilities'],
            "timestamp": timestamp,
            "event_id": db_result
        }
    
    except Exception as e:
        logger.error(f"Error in classification: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/events")
async def get_events(sound_class: str = None, limit: int = 50, offset: int = 0):
    """
    Retrieve stored events with optional filtering.
    """
    events = db.get_events(sound_class=sound_class, limit=limit, offset=offset)
    return {"events": events, "count": len(events)}

@app.get("/ui")
async def get_ui():
    """Serve simple web UI for viewing events"""
    return HTMLResponse(get_html_ui())

def parse_wav_data(wav_bytes):
    """Parse WAV file from bytes"""
    import io
    try:
        wav_file = wave.open(io.BytesIO(wav_bytes), 'rb')
        frames = wav_file.readframes(wav_file.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
        wav_file.close()
        return audio_data
    except Exception as e:
        logger.error(f"Failed to parse WAV: {e}")
        return None

def get_html_ui():
    """Simple HTML UI for viewing events - embedded inline"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sound Detection History</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                padding: 30px;
            }
            h1 { 
                color: #333; 
                margin-bottom: 10px;
                font-size: 32px;
            }
            .subtitle {
                color: #666;
                margin-bottom: 20px;
                font-size: 14px;
            }
            table { 
                border-collapse: collapse; 
                width: 100%; 
                margin-top: 20px;
            }
            th { 
                background: #667eea; 
                color: white; 
                padding: 15px; 
                text-align: left;
                font-weight: 600;
                border: none;
            }
            td { 
                border-bottom: 1px solid #eee; 
                padding: 12px 15px;
                text-align: left;
            }
            tr:hover { 
                background: #f8f9ff;
            }
            .sound-type {
                font-weight: 600;
                color: #667eea;
                text-transform: capitalize;
            }
            .confidence {
                color: #4CAF50;
                font-weight: 600;
            }
            .db-level {
                color: #FF9800;
            }
            .high-confidence {
                background: #c8e6c9;
            }
            .low-confidence {
                background: #ffcccc;
            }
            .loading {
                text-align: center;
                color: #999;
                padding: 40px;
            }
            .stats {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            .stat-card {
                flex: 1;
                min-width: 150px;
                background: #f5f7ff;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .stat-value {
                font-size: 28px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                font-size: 12px;
                color: #999;
                margin-top: 5px;
                text-transform: uppercase;
            }
            .empty {
                text-align: center;
                color: #999;
                padding: 60px 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔊 Sound Detection Dashboard</h1>
            <div class="subtitle">Real-time household sound classification system</div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value" id="total-events">0</div>
                    <div class="stat-label">Total Events</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="last-detection">-</div>
                    <div class="stat-label">Last Detection</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="avg-confidence">0%</div>
                    <div class="stat-label">Avg Confidence</div>
                </div>
            </div>
            
            <table id="events-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Sound Type</th>
                        <th>Confidence</th>
                        <th>dB Level</th>
                    </tr>
                </thead>
                <tbody id="events-body">
                    <tr class="loading"><td colspan="4">Loading events...</td></tr>
                </tbody>
            </table>
        </div>

        <script>
            async function loadEvents() {
                try {
                    const resp = await fetch('/events?limit=100');
                    const data = await resp.json();
                    
                    if (!data.events || data.events.length === 0) {
                        document.getElementById('events-body').innerHTML = 
                            '<tr class="empty"><td colspan="4">No events detected yet. Sounds will appear here.</td></tr>';
                        return;
                    }
                    
                    const tbody = document.getElementById('events-body');
                    tbody.innerHTML = '';
                    
                    let totalConfidence = 0;
                    data.events.forEach((e, idx) => {
                        const row = tbody.insertRow();
                        const confPercent = (e.confidence * 100).toFixed(1);
                        const dbLevel = e.decibel_level ? e.decibel_level.toFixed(1) : 'N/A';
                        
                        row.className = e.confidence > 0.85 ? 'high-confidence' : 
                                       e.confidence < 0.7 ? 'low-confidence' : '';
                        
                        row.innerHTML = `
                            <td>${new Date(e.timestamp).toLocaleString()}</td>
                            <td class="sound-type">${e.sound_class.replace('_', ' ')}</td>
                            <td class="confidence">${confPercent}%</td>
                            <td class="db-level">${dbLevel} dB</td>
                        `;
                        
                        totalConfidence += e.confidence;
                    });
                    
                    // Update stats
                    document.getElementById('total-events').textContent = data.events.length;
                    const lastEvent = new Date(data.events[0].timestamp);
                    const now = new Date();
                    const diff = Math.round((now - lastEvent) / 1000);
                    let timeStr = diff < 60 ? diff + 's ago' : 
                                  diff < 3600 ? Math.floor(diff/60) + 'm ago' : 
                                  lastEvent.toLocaleTimeString();
                    document.getElementById('last-detection').textContent = timeStr;
                    document.getElementById('avg-confidence').textContent = 
                        (totalConfidence / data.events.length * 100).toFixed(0) + '%';
                    
                } catch (err) {
                    console.error('Error loading events:', err);
                    document.getElementById('events-body').innerHTML = 
                        '<tr class="loading"><td colspan="4">Error loading events. Server may be down.</td></tr>';
                }
            }
            
            // Initial load
            loadEvents();
            
            // Refresh every 3 seconds
            setInterval(loadEvents, 3000);
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
