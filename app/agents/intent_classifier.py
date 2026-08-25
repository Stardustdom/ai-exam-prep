# app/agents/intent_classifier.py
#
# Recognizes "exit"/"start" INTENT from free text that doesn't literally
# match a fixed phrase list — "good bye", "cya", "howdy" — via a guarded LLM
# classification, cached the same way exam/chapter resolution already is
# (app.services.semantic_cache): once a specific phrase has been classified,
# every later occurrence of it is free. Two layers, cheapest first:
#
#   1. Exact-match against a hardcoded set — the overwhelming majority of
#      real traffic ("bye", "hi", "start", ...), zero API calls ever.
#   2. Role-prompted LLM classification, ONLY for messages short enough and
#      ambiguous enough to plausibly be a greeting/farewell rather than a
#      real answer — strictly validated to one of three labels, defaulting
#      to OTHER on any doubt or malformed output, so a misfire can never
#      accidentally end or restart a real session.
#
# Callers are responsible for not invoking this while a menu (exam/chapter
# selection) is actively expecting a pick — see app.bot.handlers — since a
# short, single-word curriculum name (e.g. "Sets") is exactly the shape of
# input this classifier is designed to catch, and only the caller knows
# whether that context applies.
from typing import Literal
from app.services.llm import LLMService
from app.services.semantic_cache import SemanticCacheService
import logging

logger = logging.getLogger(__name__)

Intent = Literal["exit", "start", "other"]

EXIT_PHRASES = {
    "bye", "goodbye", "good bye", "bye bye", "see you", "see ya", "cya",
    "quit", "exit", "end", "stop", "cancel", "/end", "later", "gtg",
    "i'm done", "im done", "done", "leave", "gotta go", "got to go",
    "talk later", "ttyl", "peace out", "im out", "i'm out",
}
START_PHRASES = {
    "start", "hi", "hello", "hey", "yo", "begin", "hiya", "sup",
    "/start", "restart", "hey there", "hi there", "howdy", "yo yo",
    "whats up", "what's up",
}

_ROLE_PROMPT = """You are an intent classifier for a Telegram exam-prep bot. \
The user's message arrives mid-conversation with a quiz bot, so it is very \
likely a real answer — an exam name, a chapter/subject/topic name (these \
are often a single short word, e.g. "Sets", "Probability", "Matrices"), a \
number, or similar. Only rarely does it instead express one of two special \
intents:

- EXIT: the user clearly wants to end/stop/quit the conversation (e.g. "gotta run", "not now", "talk to you later")
- START: the user is clearly just greeting the bot, or asking to begin/restart with no other content (e.g. "yo", "hiya")

Respond with EXACTLY one word: EXIT, START, or OTHER.

Examples:
"good bye" -> EXIT
"gotta run" -> EXIT
"hiya" -> START
"yo" -> START
"Sets" -> OTHER
"Probability" -> OTHER
"20" -> OTHER
"JEE" -> OTHER
"Physics" -> OTHER

If there is ANY doubt, or the message could plausibly be a real answer, \
respond OTHER. Never guess EXIT or START."""


async def classify_intent(
    text: str, llm_service: LLMService, cache_service: SemanticCacheService
) -> Intent:
    normalized = text.strip().lower()
    if not normalized:
        return "other"

    if normalized in EXIT_PHRASES:
        return "exit"
    if normalized in START_PHRASES:
        return "start"

    # A real exam/chapter name or a numeric answer should never reach the
    # LLM call below — cheap guards before paying for a classification.
    word_count = len(normalized.split())
    is_numeric = normalized.replace(".", "", 1).replace("-", "", 1).isdigit()
    if word_count > 3 or is_numeric:
        return "other"

    cached = await cache_service.get_resolved_entity(query=normalized, entity_type="message_intent")
    if cached:
        entity_id = cached.get("entity_id")
        if entity_id in ("exit", "start", "other"):
            return entity_id  # type: ignore[return-value]

    intent: Intent = "other"
    try:
        response = await llm_service.generate_text(f"{_ROLE_PROMPT}\n\nMessage: \"{text}\"", temperature=0.0)
        label = response.strip().upper()
        if label == "EXIT":
            intent = "exit"
        elif label == "START":
            intent = "start"
        # anything else (including "OTHER" itself, or a malformed response) stays "other"
    except Exception as e:
        logger.warning(f"Intent classification failed, defaulting to 'other': {e}")

    await cache_service.cache_resolved_entity(
        query=normalized, entity_type="message_intent", entity_id=intent,
        confidence=1.0, ttl_hours=24 * 30
    )
    return intent


_OFF_TOPIC_ROLE_PROMPT = """You are helping an exam-prep Telegram bot give a \
better error message. The user's message FAILED to match any real exam or \
chapter/topic name via exact and semantic matching - your only job is to \
judge whether it's clearly unrelated small talk / a question about the bot \
itself (so the bot should say "I'm just a study bot" instead of "try \
again"), or whether it still looks like a genuine (if unsuccessful) attempt \
to name an exam or chapter (a typo, an unusual phrasing, a real subject the \
corpus just doesn't have).

Respond with EXACTLY one word: YES (clearly off-topic/small-talk) or NO \
(still looks like a genuine attempt). If there is ANY doubt, respond NO.

Examples:
"whats your name" -> YES
"who made you" -> YES
"how are you" -> YES
"lol" -> YES
"phyiscs" -> NO
"organic chem" -> NO
"idk maybe calculus" -> NO
"quantum mechanics" -> NO"""


async def is_off_topic(
    text: str, llm_service: LLMService, cache_service: SemanticCacheService
) -> bool:
    """Only meaningful to call from a resolution FAILURE path (exam/chapter
    matching has already tried and failed) — never gates whether matching is
    attempted in the first place, so it can't collide with or override a
    real selection. Same cost/safety layering as classify_intent: cached
    after the first classification of any given phrase, and defaults to
    False (i.e. "still treat it as a genuine attempt") on any doubt, cache
    miss requiring an LLM call, or classification failure — the safe
    default keeps today's existing "couldn't find that, try again" message
    rather than risk telling a real (if unsuccessful) attempt "I'm just a
    study bot"."""
    normalized = text.strip().lower()
    if not normalized or len(normalized.split()) > 8:
        return False

    cached = await cache_service.get_resolved_entity(query=normalized, entity_type="off_topic_check")
    if cached:
        return cached.get("entity_id") == "yes"

    result = False
    try:
        response = await llm_service.generate_text(f"{_OFF_TOPIC_ROLE_PROMPT}\n\nMessage: \"{text}\"", temperature=0.0)
        result = response.strip().upper() == "YES"
    except Exception as e:
        logger.warning(f"Off-topic classification failed, defaulting to False: {e}")

    await cache_service.cache_resolved_entity(
        query=normalized, entity_type="off_topic_check", entity_id=("yes" if result else "no"),
        confidence=1.0, ttl_hours=24 * 30
    )
    return result
