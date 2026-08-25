from telegram import Bot, InlineKeyboardMarkup
from telegram.constants import ParseMode
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TelegramService:
    def __init__(self):
        self.bot = Bot(token=settings.telegram_bot_token)

    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_markup: InlineKeyboardMarkup = None,
        parse_mode: str = ParseMode.HTML
    ):
        """Send a message to a Telegram chat"""
        try:
            return await self.bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=reply_markup,
                parse_mode=parse_mode
            )
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None

    async def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: InlineKeyboardMarkup = None
    ):
        """Edit an existing message"""
        try:
            return await self.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                reply_markup=reply_markup
            )
        except Exception as e:
            logger.error(f"Failed to edit message: {e}")
            return None