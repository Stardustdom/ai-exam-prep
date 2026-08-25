# app/bot/handlers.py
import asyncio
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ContextTypes
from sqlalchemy import text
from app.database import AsyncSessionLocal
from app.bot.dependencies import build_context
from app.agents.state import ExamSessionState, SessionStep
from app.graph.checkpointer import is_connection_error, reinit_checkpointer
import logging

logger = logging.getLogger(__name__)

# Serializes ALL processing (classification, graph resume/start, exit) per
# chat — confirmed necessary by direct reproduction: two messages for the
# SAME chat processed concurrently (e.g. a reply-keyboard tap landing right
# after a slow LLM-backed classification of the previous message) both read
# the same starting checkpoint and each write back independently; whichever
# finishes last silently wins, discarding the other's real result even
# though ITS OWN response already told the user it succeeded. A single
# Render instance (WEB_CONCURRENCY=1) still runs requests concurrently via
# asyncio, so an in-process lock is sufficient here — it would NOT be
# sufficient across multiple instances, which this deployment doesn't have.
_chat_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


def _chat_lock(chat_id) -> asyncio.Lock:
    return _chat_locks[str(chat_id)]


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # /start always means "give me a clean slate", including when the user is
    # stuck mid-conversation (e.g. a chapter that wouldn't resolve) — it must
    # NOT be fed as if it were their answer to whatever step is currently
    # paused (previously: literally tried to resolve "start" as a chapter
    # name / question count / etc., which just produced another confusing
    # error instead of the reset the user was asking for).
    async with _chat_lock(update.effective_chat.id):
        await _handle_turn(update, context, "start", force_restart=True)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    async with _chat_lock(update.effective_chat.id):
        intent = await _classify_message_intent(update.message.text or "", update)
        if intent == "exit":
            await _handle_exit(update, context)
            return
        if intent == "start":
            # A greeting ("hi"/"hello"/...) means the same thing /start does —
            # give a clean slate, mid-conversation or not.
            await _handle_turn(update, context, "start", force_restart=True)
            return
        await _handle_turn(update, context, update.message.text)


# Steps where free text IS the answer being evaluated (a real curriculum
# name can legitimately be one short word — "Sets", "Probability"). The
# LLM-backed fallback in classify_intent is specifically shaped to catch
# short, ambiguous phrases, which makes it also the shape most likely to
# collide with a real selection here — so it's skipped entirely while one
# of these menus is active. The hardcoded exact-match phrases (a literal
# "bye" or "hi") still apply even then, since no real curriculum entry
# collides with those specific words.
_MENU_ACTIVE_STEPS = {SessionStep.SELECT_EXAM.value, SessionStep.SELECT_CHAPTER.value}


async def _classify_message_intent(raw_text: str, update: Update) -> str:
    from app.agents.intent_classifier import classify_intent, EXIT_PHRASES, START_PHRASES

    normalized = raw_text.strip().lower()
    if normalized in EXIT_PHRASES:
        return "exit"
    if normalized in START_PHRASES:
        return "start"
    if not normalized or len(normalized.split()) > 3:
        return "other"  # cheap guard, matches classify_intent's own — skip opening a session at all

    telegram_user = update.effective_user
    chat_id = update.effective_chat.id
    try:
        async with AsyncSessionLocal() as db:
            ctx = build_context(db)
            state = ExamSessionState(telegram_chat_id=str(chat_id))
            state = await ctx.session_manager.initialize_session(
                state, telegram_user_id=str(telegram_user.id), telegram_chat_id=str(chat_id),
                username=telegram_user.username, first_name=telegram_user.first_name, last_name=telegram_user.last_name
            )
            if state.error:
                return "other"

            prompt = await ctx.graph.get_prompt(state.session_id)
            if (prompt or {}).get("step") in _MENU_ACTIVE_STEPS:
                return "other"

            from app.services.embeddings import EmbeddingService
            from app.services.semantic_cache import SemanticCacheService
            from app.services.llm import LLMService
            from app.database.repositories import SemanticCacheRepository
            cache_service = SemanticCacheService(SemanticCacheRepository(db), EmbeddingService())
            return await classify_intent(raw_text, LLMService(), cache_service)
    except Exception as e:
        # Best-effort enhancement, not core functionality — any failure here
        # (including the checkpointer's known Neon-connection-death class of
        # error, which _run_graph_turn recovers from but this path doesn't
        # duplicate that retry logic for) must fall back to normal message
        # processing, never block or crash on the user's actual message.
        logger.warning(f"Intent classification pre-check failed for chat {chat_id}, defaulting to 'other': {e}")
        return "other"


async def _handle_exit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """"bye"/"exit"/etc. must end the session then and there — NOT be fed as
    the answer to whatever step happens to be active (a chapter attempt, a
    duration, ...), and NOT immediately re-prompt with a fresh exam-selection
    screen the way /start's reset does. So this reuses the same reset
    machinery (clear the DB session row, drop the live-timer job, purge the
    thread's LangGraph checkpoint) but deliberately never calls graph.start()
    afterward — the conversation just stops, cleanly, until the user chooses
    to begin again."""
    telegram_user = update.effective_user
    chat_id = update.effective_chat.id

    async with AsyncSessionLocal() as db:
        ctx = build_context(db)
        state = ExamSessionState(telegram_chat_id=str(chat_id), current_step=SessionStep.START)
        state = await ctx.session_manager.initialize_session(
            state,
            telegram_user_id=str(telegram_user.id),
            telegram_chat_id=str(chat_id),
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name,
            reset=True
        )
        if state.error:
            logger.error(f"Session init failed during exit for chat {chat_id}: {state.error}")
            return

        thread_id = state.session_id
        _cancel_timer_tick(context, thread_id)
        await _purge_thread_checkpoint(ctx.db, thread_id)

        await ctx.telegram_service.send_message(
            chat_id=chat_id, text="👋 Session ended. Send /start anytime to begin a new one.",
            reply_markup=ReplyKeyboardRemove()
        )


async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()  # ack immediately, outside the lock — unrelated to our own processing
    data = query.data

    async with _chat_lock(update.effective_chat.id):
        # The results-screen buttons operate on a COMPLETED thread (the graph
        # has already reached END) — none of the interrupt/resume machinery
        # applies to them. Previously they had no handler at all, so pressing
        # any of them fell through to the generic flow, which sees "not
        # awaiting input" and calls graph.start() fresh — silently generating
        # a whole NEW quiz using the old leftover exam/chapter/duration
        # instead of doing what the button actually says.
        if data == "review_answers":
            await _handle_review_answers(update, context)
            return
        if data == "view_explanations":
            await _handle_view_explanations(update, context)
            return
        if data in ("take_another", "main_menu"):
            await _handle_turn(update, context, "start", force_restart=True)
            return

        await _handle_turn(update, context, data, is_callback=True)


async def _handle_turn(
    update: Update, context: ContextTypes.DEFAULT_TYPE, raw_input: str,
    is_callback: bool = False, force_restart: bool = False
) -> None:
    """
    Every Telegram update — text message or button click — funnels through
    here: load/create the session, resolve whether the graph thread is fresh
    or mid-conversation, advance it exactly one turn, and render whatever it
    pauses on next.
    """
    telegram_user = update.effective_user
    chat_id = update.effective_chat.id

    async with AsyncSessionLocal() as db:
        ctx = build_context(db)

        state = ExamSessionState(telegram_chat_id=str(chat_id), current_step=SessionStep.START)
        state = await ctx.session_manager.initialize_session(
            state,
            telegram_user_id=str(telegram_user.id),
            telegram_chat_id=str(chat_id),
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name,
            reset=force_restart
        )
        if state.error:
            await ctx.telegram_service.send_message(chat_id=chat_id, text="Something went wrong starting your session. Please try again.")
            logger.error(f"Session init failed for chat {chat_id}: {state.error}")
            return

        thread_id = state.session_id

        try:
            result = await _run_graph_turn(db, chat_id, state, thread_id, raw_input, is_callback, force_restart, context)
        except Exception as e:
            logger.error(f"Graph execution failed for chat {chat_id}: {e}")
            await ctx.telegram_service.send_message(
                chat_id=chat_id,
                text="Sorry, something went wrong. Please try again."
            )
            return

        await _render(ctx, chat_id, result, update, context, thread_id)


async def _run_graph_turn(
    db, chat_id: int, state: ExamSessionState, thread_id: str,
    raw_input: str, is_callback: bool, force_restart: bool,
    context: ContextTypes.DEFAULT_TYPE
) -> Dict[str, Any]:
    """Runs is_awaiting_input -> resolve_input -> graph.start/resume,
    retrying once with a freshly reinitialized checkpointer if it fails on
    what looks like a dead Neon connection. AsyncPostgresSaver (the
    checkpointer) opens one long-lived connection at startup with no
    equivalent to app.database's pool_pre_ping — Neon closing that
    connection (free-tier idle suspend, or an admin-initiated restart even
    on an active project) previously left every /start and every quiz
    action failing with a generic "something went wrong" until someone
    manually restarted the whole Render service. build_context() is cheap
    (just wires up repos/agents against the existing db session and
    whatever checkpointer get_checkpointer() currently returns), so
    rebuilding it each attempt costs nothing but picks up the reinitialized
    checkpointer on retry."""
    for attempt in range(2):
        ctx = build_context(db)
        try:
            if force_restart:
                _cancel_timer_tick(context, thread_id)  # abandon any in-progress quiz's live-timer job
                await _purge_thread_checkpoint(ctx.db, thread_id)  # drop any paused/pending task for this thread
                awaiting = False
                graph_input = None
            else:
                awaiting = await ctx.graph.is_awaiting_input(thread_id)
                graph_input = await _resolve_graph_input(ctx, thread_id, raw_input, is_callback)

            if awaiting:
                return await ctx.graph.resume(thread_id, graph_input)
            return await ctx.graph.start(state)
        except Exception as e:
            if attempt == 0 and is_connection_error(e):
                logger.warning(f"Checkpointer connection error for chat {chat_id}, reinitializing and retrying once: {e}")
                await reinit_checkpointer()
                continue
            raise


async def _purge_thread_checkpoint(db, thread_id: str) -> None:
    """A real /start reset needs the thread's LangGraph checkpoint history
    gone, not just the ExamSessionState fields cleared — proven empirically:
    ainvoke()-ing a brand-new, fully-cleared state at a thread_id that still
    has a paused/pending task silently ignored that new state and returned
    stale interrupt data from the old task instead. LangGraph's own
    BaseCheckpointSaver.adelete_thread() would be the proper API for this,
    but AsyncPostgresSaver in the installed langgraph version (0.2.62)
    doesn't implement it (raises NotImplementedError) — so delete directly
    from its three per-thread tables instead, reusing the same DB session
    the rest of this request already has open."""
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        await db.execute(text(f"DELETE FROM {table} WHERE thread_id = :tid"), {"tid": thread_id})
    await db.commit()


async def _resolve_graph_input(ctx, thread_id: str, raw_input: str, is_callback: bool) -> Any:
    """
    Turns a raw Telegram callback_data string into whatever the graph node at
    the current interrupt actually expects (a plain string for exam/chapter
    selection and free text, or a structured action dict for in-quiz button
    presses, since Telegram callback_data can't itself carry option text —
    it's capped at 64 bytes).
    """
    if not is_callback:
        return raw_input

    if raw_input.startswith("ans_"):
        _, q_idx, opt_idx = raw_input.split("_", 2)
        prompt = await ctx.graph.get_prompt(thread_id) or {}
        questions = prompt.get("questions") or []
        q_idx, opt_idx = int(q_idx), int(opt_idx)
        answer_text = questions[q_idx]["options"][opt_idx] if q_idx < len(questions) else opt_idx
        return {"type": "answer", "question_index": q_idx, "answer": answer_text}

    if raw_input.startswith("nav_"):
        return {"type": "nav", "index": int(raw_input.split("_", 1)[1])}

    if raw_input == "submit_quiz":
        return {"type": "submit"}

    # exam_<id>, chapter_<id|overall>, start_quiz, or raw free text — the graph
    # nodes themselves recognize the exam_/chapter_ prefixes and "start_quiz".
    return raw_input


# ---- rendering --------------------------------------------------------------

async def _render(
    ctx, chat_id: int, result: Dict[str, Any],
    update: Optional[Update] = None, context: Optional[ContextTypes.DEFAULT_TYPE] = None,
    thread_id: Optional[str] = None
) -> None:
    prompt = result.get("prompt")
    if prompt is not None:
        await _render_prompt(ctx, chat_id, prompt, update, context, thread_id)
        return

    state = result.get("state") or {}
    if state.get("evaluation"):
        await _render_results(ctx, chat_id, state, update, context, thread_id)
    else:
        # Graph reached END without an evaluation (shouldn't normally happen)
        await ctx.telegram_service.send_message(chat_id=chat_id, text="Session ended. Send any message to start again.")


async def _render_prompt(
    ctx, chat_id: int, prompt: Dict[str, Any],
    update: Optional[Update], context: Optional[ContextTypes.DEFAULT_TYPE], thread_id: Optional[str]
) -> None:
    step = prompt.get("step")

    if step == SessionStep.SELECT_EXAM.value:
        # Reply keyboard, not inline: an inline button press never appears as
        # a message in the chat at all (just a silent callback) — a reply
        # keyboard button sends its label as a genuine text message from the
        # user, which _resolve_exam_node already resolves correctly via its
        # existing exact-name-match step (the same path free-typing the exam
        # name has always used), so no graph-side change is needed here.
        exams = await ctx.exam_resolver.get_all_active_exams()
        buttons = [[e["name"]] for e in exams]
        await ctx.telegram_service.send_message(
            chat_id=chat_id, text=prompt["message"],
            reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=True) if buttons else None
        )

    elif step == SessionStep.SELECT_QUESTION_COUNT.value:
        # Explicitly clear the exam-selection reply keyboard now that it's
        # served its purpose — one_time_keyboard alone doesn't reliably hide
        # it (it never auto-showed on desktop/web in the first place, so
        # there's nothing there for that flag to hide; it would otherwise
        # just sit behind the keyboard icon indefinitely).
        await ctx.telegram_service.send_message(
            chat_id=chat_id, text=prompt["message"], reply_markup=ReplyKeyboardRemove()
        )

    elif step == SessionStep.SELECT_CHAPTER.value:
        options = prompt.get("options") or []
        buttons = [["Overall"]] + [[opt["name"]] for opt in options[:30]]
        await ctx.telegram_service.send_message(
            chat_id=chat_id, text=prompt["message"],
            reply_markup=ReplyKeyboardMarkup(buttons, resize_keyboard=True, one_time_keyboard=True)
        )

    elif step == SessionStep.SELECT_DURATION.value:
        # Clear the chapter-selection keyboard — see SELECT_QUESTION_COUNT above.
        await ctx.telegram_service.send_message(
            chat_id=chat_id, text=prompt["message"], reply_markup=ReplyKeyboardRemove()
        )

    elif step == SessionStep.QUIZ_READY.value:
        await ctx.telegram_service.send_message(
            chat_id=chat_id, text=prompt["message"],
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("▶ Start Quiz", callback_data="start_quiz")]])
        )

    elif step == SessionStep.QUIZ_IN_PROGRESS.value:
        await _render_question(ctx, chat_id, prompt, update, context, thread_id)


def _format_question(
    questions: List[Dict[str, Any]], idx: int, answers: Dict[Any, str], expiry_time: Optional[str]
) -> Tuple[str, InlineKeyboardMarkup]:
    """Pure formatter shared by the interactive render path and the live-timer
    ticker below, so the two never drift out of sync with each other."""
    question = questions[idx]

    answered = len(answers)
    text = f"Question {idx + 1} / {len(questions)}  (Answered: {answered}/{len(questions)})\n\n"
    text += question.get("question_text", "") + "\n\n"
    options = question.get("options", [])
    for i, option in enumerate(options):
        # answers round-trips through JSON (ChatSession.state_data), which
        # turns int keys into strings — check both.
        marker = "💠 " if answers.get(str(idx)) == option or answers.get(idx) == option else ""
        text += f"{marker}{chr(65 + i)}. {option}\n"

    if expiry_time:
        remaining = datetime.fromisoformat(expiry_time) - datetime.utcnow()
        total_seconds = max(int(remaining.total_seconds()), 0)
        text += f"\n⏱ Time Left: {total_seconds // 60:02d}:{total_seconds % 60:02d}"

    buttons = [
        [InlineKeyboardButton(chr(65 + i), callback_data=f"ans_{idx}_{i}") for i in range(len(options))]
    ]
    nav_row = []
    if idx > 0:
        nav_row.append(InlineKeyboardButton("◀ Previous", callback_data=f"nav_{idx - 1}"))
    if idx < len(questions) - 1:
        nav_row.append(InlineKeyboardButton("Next ▶", callback_data=f"nav_{idx + 1}"))
    if nav_row:
        buttons.append(nav_row)
    buttons.append([InlineKeyboardButton("📤 Submit Test", callback_data="submit_quiz")])

    return text, InlineKeyboardMarkup(buttons)


async def _render_question(
    ctx, chat_id: int, prompt: Dict[str, Any],
    update: Optional[Update], context: Optional[ContextTypes.DEFAULT_TYPE], thread_id: Optional[str]
) -> None:
    """Google-Forms-style single-message quiz: every question, answer, and
    nav press edits the SAME Telegram message in place (it always originates
    from a button press on that message, so its message_id is always
    available via the callback) instead of sending a new message each time."""
    questions = prompt.get("questions") or []
    idx = prompt.get("current_index", 0)
    answers = prompt.get("answers") or {}
    if not questions or idx >= len(questions):
        return
    expiry_time = prompt.get("expiry_time")

    text, markup = _format_question(questions, idx, answers, expiry_time)

    message_id = None
    if update is not None and update.callback_query is not None:
        message_id = update.callback_query.message.message_id
        await ctx.telegram_service.edit_message(chat_id=chat_id, message_id=message_id, text=text, reply_markup=markup)
    else:
        # First-ever render of this quiz didn't come from a button (shouldn't
        # normally happen, since even "Start Quiz" is a callback) — send fresh.
        sent = await ctx.telegram_service.send_message(chat_id=chat_id, text=text, reply_markup=markup)
        message_id = sent.message_id if sent else None

    if message_id:
        _schedule_timer_tick(context, chat_id, message_id, thread_id)


async def _render_results(
    ctx, chat_id: int, state: Dict[str, Any],
    update: Optional[Update], context: Optional[ContextTypes.DEFAULT_TYPE], thread_id: Optional[str]
) -> None:
    _cancel_timer_tick(context, thread_id)

    evaluation = state.get("evaluation") or {}

    text = "🎯 TEST COMPLETED\n\n"
    text += f"Exam: {state.get('exam_name')}\n"
    text += f"Chapter: {state.get('chapter_name') or 'Overall'}\n\n"
    text += f"📊 Questions: {evaluation.get('total_questions', 0)}\n"
    text += f"✅ Attempted: {evaluation.get('attempted', 0)}\n"
    text += f"✓ Correct: {evaluation.get('correct', 0)}\n"
    text += f"✗ Incorrect: {evaluation.get('incorrect', 0)}\n"
    text += f"⏭ Unanswered: {evaluation.get('unanswered', 0)}\n\n"
    text += f"📈 Score: {evaluation.get('score', 0)}\n"
    text += f"🎯 Accuracy: {evaluation.get('accuracy', 0):.1f}%\n\n"

    if evaluation.get("time_taken_seconds"):
        mins, secs = divmod(int(evaluation["time_taken_seconds"]), 60)
        text += f"⏱ Time Used: {mins}m {secs}s\n\n"

    performance = evaluation.get("performance", {}) or {}
    strong_areas = performance.get("strong_areas", [])
    weak_areas = performance.get("weak_areas", [])
    if strong_areas:
        text += "💪 Strong Areas:\n" + "".join(f"• {a}\n" for a in strong_areas[:5]) + "\n"
    if weak_areas:
        text += "📚 Needs Improvement:\n" + "".join(f"• {a}\n" for a in weak_areas[:5]) + "\n"

    buttons = [
        [InlineKeyboardButton("📝 Review Answers", callback_data="review_answers")],
        [InlineKeyboardButton("📖 View Explanations", callback_data="view_explanations")],
        [InlineKeyboardButton("🔄 Take Another Test", callback_data="take_another")],
        [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")],
    ]
    markup = InlineKeyboardMarkup(buttons)

    if update is not None and update.callback_query is not None:
        # Submit was itself a button press on the question message — finish
        # the "one message" flow by turning that same message into the
        # results screen rather than sending a separate one.
        await ctx.telegram_service.edit_message(
            chat_id=chat_id, message_id=update.callback_query.message.message_id, text=text, reply_markup=markup
        )
    else:
        await ctx.telegram_service.send_message(chat_id=chat_id, text=text, reply_markup=markup)


async def _get_latest_quiz(ctx, chat_id: int, telegram_user) -> Optional[Any]:
    """The quiz this chat's session most recently completed (or is on)."""
    user = await ctx.user_repo.get_or_create(telegram_user_id=str(telegram_user.id))
    chat_session = await ctx.chat_session_repo.get_or_create(user_id=str(user.id), telegram_chat_id=str(chat_id))
    quiz_id = (chat_session.state_data or {}).get("quiz_id")
    return await ctx.quiz_repo.get_by_id(quiz_id) if quiz_id else None


async def _handle_review_answers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows each question next to the user's answer and the correct one."""
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as db:
        ctx = build_context(db)
        quiz = await _get_latest_quiz(ctx, chat_id, update.effective_user)

        if not quiz or not quiz.generated_questions:
            await ctx.telegram_service.send_message(chat_id=chat_id, text="No recent test found to review.")
            return

        questions = quiz.generated_questions
        answers = {a["question_index"]: a["answer"] for a in (quiz.user_answers or [])}

        lines = ["REVIEW ANSWERS", ""]
        for idx, q in enumerate(questions):
            user_answer = answers.get(idx)
            correct = q.get("correct_answer")
            if user_answer is None:
                marker = "⏭"
            elif user_answer == correct:
                marker = "✅"
            else:
                marker = "❌"
            lines.append(f"{marker} Q{idx + 1}. {q.get('question_text', '')}")
            lines.append(f"Your answer: {user_answer or '(unanswered)'}")
            if user_answer != correct:
                lines.append(f"Correct answer: {correct}")
            lines.append("")

        # parse_mode=None: question text is LLM-generated and may contain
        # raw <, >, & (e.g. inequalities) that would otherwise break Telegram's
        # HTML parser and silently drop the message.
        for chunk in _chunk_lines(lines):
            await ctx.telegram_service.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


async def _handle_view_explanations(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Full worked solution for every question. The data isn't generated
    fresh here — QuestionGeneratorAgent already asks the LLM for an
    "explanation" (grounded in the retrieved study-material chunks, per its
    prompt) and a "source_reference" for every question at quiz-creation
    time, and both are already sitting in quiz.generated_questions unused.
    This button was a placeholder for no real reason; wiring it up needs no
    new LLM calls, no new rate-limit exposure — just displaying what's
    already there."""
    chat_id = update.effective_chat.id
    async with AsyncSessionLocal() as db:
        ctx = build_context(db)
        quiz = await _get_latest_quiz(ctx, chat_id, update.effective_user)

        if not quiz or not quiz.generated_questions:
            await ctx.telegram_service.send_message(chat_id=chat_id, text="No recent test found to explain.")
            return

        questions = quiz.generated_questions
        answers = {a["question_index"]: a["answer"] for a in (quiz.user_answers or [])}

        lines = ["📖 EXPLANATIONS", ""]
        for idx, q in enumerate(questions):
            user_answer = answers.get(idx)
            correct = q.get("correct_answer")
            marker = "⏭" if user_answer is None else ("✅" if user_answer == correct else "❌")
            explanation = q.get("explanation")

            lines.append(f"{marker} Q{idx + 1}. {q.get('question_text', '')}")
            lines.append(f"Answer: {correct}")
            if explanation:
                lines.append(f"Why: {explanation}")
            source = q.get("source_reference")
            if source:
                lines.append(f"Source: {source}")
            lines.append("")

        for chunk in _chunk_lines(lines):
            await ctx.telegram_service.send_message(chat_id=chat_id, text=chunk, parse_mode=None)


def _chunk_lines(lines: List[str], limit: int = 3500) -> List[str]:
    """Telegram caps a single message at 4096 chars — a quiz with enough
    questions can exceed that, so split on line boundaries into chunks."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        if current and current_len + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---- live timer --------------------------------------------------------------
#
# Button presses already show a freshly-computed "Time Left" on every edit,
# but if the user just sits reading a question nothing re-renders and the
# countdown looks frozen. This repeating job keeps it moving in the
# background. It's scheduled per-quiz (python-telegram-bot's JobQueue, an
# in-process APScheduler) and always re-reads current state from the DB on
# each tick rather than closing over a snapshot, since the user may answer or
# navigate between ticks. It self-cancels once the quiz leaves
# QUIZ_IN_PROGRESS or its countdown hits zero. Being in-process, a scheduled
# tick is lost across an app restart mid-quiz — cosmetic only, since the
# graph's own state stays safely checkpointed in Postgres regardless; the
# next button press naturally reschedules it.

TICK_INTERVAL_SECONDS = 20


def _timer_job_name(thread_id: str) -> str:
    return f"quiz-timer-{thread_id}"


def _schedule_timer_tick(
    context: Optional[ContextTypes.DEFAULT_TYPE], chat_id: int, message_id: int, thread_id: Optional[str]
) -> None:
    if context is None or context.job_queue is None or not thread_id:
        return
    name = _timer_job_name(thread_id)
    for job in context.job_queue.get_jobs_by_name(name):
        job.schedule_removal()
    context.job_queue.run_repeating(
        _tick_quiz_timer,
        interval=TICK_INTERVAL_SECONDS,
        first=TICK_INTERVAL_SECONDS,
        name=name,
        data={"chat_id": chat_id, "message_id": message_id, "thread_id": thread_id}
    )


def _cancel_timer_tick(context: Optional[ContextTypes.DEFAULT_TYPE], thread_id: Optional[str]) -> None:
    if context is None or context.job_queue is None or not thread_id:
        return
    for job in context.job_queue.get_jobs_by_name(_timer_job_name(thread_id)):
        job.schedule_removal()


async def _tick_quiz_timer(context: ContextTypes.DEFAULT_TYPE) -> None:
    data = context.job.data
    chat_id, message_id, thread_id = data["chat_id"], data["message_id"], data["thread_id"]

    async with AsyncSessionLocal() as db:
        ctx = build_context(db)
        chat_session = await ctx.chat_session_repo.get_by_id(thread_id)
        if not chat_session or chat_session.current_step != SessionStep.QUIZ_IN_PROGRESS.value:
            context.job.schedule_removal()
            return

        state_data = chat_session.state_data or {}
        quiz_id = state_data.get("quiz_id")
        expiry_time = state_data.get("expiry_time")
        idx = state_data.get("current_question_index", 0)
        answers = state_data.get("answers") or {}

        quiz = await ctx.quiz_repo.get_by_id(quiz_id) if quiz_id else None
        questions = (quiz.generated_questions if quiz else None) or []
        if not questions or idx >= len(questions):
            context.job.schedule_removal()
            return

        text, markup = _format_question(questions, idx, answers, expiry_time)
        await ctx.telegram_service.edit_message(chat_id=chat_id, message_id=message_id, text=text, reply_markup=markup)

        if expiry_time:
            remaining = datetime.fromisoformat(expiry_time) - datetime.utcnow()
            if remaining.total_seconds() <= 0:
                # Countdown has visibly hit 00:00 — the expiry sweep (or the
                # user's next interaction) handles the actual submit; the
                # ticker's only job was the visible countdown, so stop here.
                context.job.schedule_removal()
