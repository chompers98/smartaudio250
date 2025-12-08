import logging
from twilio.rest import Client
from config import TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_FROM, NOTIFY_PHONE_TO

logger = logging.getLogger(__name__)

class NotificationManager:
    """Send notifications via SMS"""
    
    def __init__(self):
        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            self.client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        else:
            self.client = None
            logger.warning("Twilio credentials not configured. SMS notifications disabled.")
    
    def send_sms(self, sound_class, confidence, decibel_level):
        """
        Send SMS notification of detected sound.
        """
        if not self.client or not NOTIFY_PHONE_TO:
            logger.info("SMS notification skipped (not configured)")
            return False
        
        message = (
            f"🔔 Sound Detected: {sound_class.replace('_', ' ').title()}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Level: {decibel_level:.1f} dB"
        )
        
        try:
            self.client.messages.create(
                body=message,
                from_=TWILIO_PHONE_FROM,
                to=NOTIFY_PHONE_TO
            )
            logger.info(f"SMS sent: {sound_class}")
            return True
        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            return False
