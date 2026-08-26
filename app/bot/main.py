from fastapi import FastAPI, Request, Response, Header, HTTPException
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ChatMemberHandler, filters
from app.config.settings import settings
from app.bot import handlers
import logging
import secrets

logger = logging.getLogger(__name__)

bot_app = FastAPI()

application: Application = None
# Telegram calls back with this secret header when set via set_webhook(secret_token=...),
# so the webhook endpoint can reject requests that didn't actually come from Telegram.
_webhook_secret = secrets.token_urlsafe(32)


def init_bot() -> Application:
    """Build the python-telegram-bot Application once at startup. Handler functions
    build their own DB session/agents per update (see app.bot.dependencies) rather
    than closing over pre-injected instances, since each update needs a fresh
    async session."""
    global application
    application = Application.builder().token(settings.telegram_bot_token).build()
    application.add_handler(CommandHandler("start", handlers.start_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handlers.message_handler))
    application.add_handler(CallbackQueryHandler(handlers.callback_handler))
    # Group auto-onboarding (Gen-Z SRS FR-1): fires when the bot's OWN status
    # in a chat changes (added/removed/promoted) — distinct from
    # CHAT_MEMBER, which is about other members and requires separately
    # opting in via allowed_updates. MY_CHAT_MEMBER is in Telegram's default
    # allowed_updates set already, so no webhook/polling config change needed.
    application.add_handler(ChatMemberHandler(handlers.my_chat_member_handler, ChatMemberHandler.MY_CHAT_MEMBER))
    return application


@bot_app.post("/telegram")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str = Header(default=None)):
    """Handle Telegram webhook requests"""
    if x_telegram_bot_api_secret_token != _webhook_secret:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")
    if application is None:
        raise HTTPException(status_code=503, detail="Bot not initialized")

    try:
        body = await request.json()
        update = Update.de_json(body, application.bot)
        await application.process_update(update)
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        # Telegram retries on non-2xx; we've already logged the failure, so
        # acknowledge receipt rather than trigger a redelivery storm.
        return Response(status_code=200)


async def setup_webhook() -> None:
    """Register the webhook URL with Telegram (production; local dev can poll instead)"""
    app = init_bot()
    await app.initialize()
    webhook_url = f"{settings.telegram_webhook_url}/webhook/telegram"
    await app.bot.set_webhook(webhook_url, secret_token=_webhook_secret)
    logger.info(f"Telegram webhook set to: {webhook_url}")


async def run_polling() -> None:
    """Local dev alternative to a public webhook URL"""
    app = init_bot()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    logger.info("Telegram bot polling started")


async def shutdown_bot() -> None:
    if application is not None:
        if application.updater and application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
