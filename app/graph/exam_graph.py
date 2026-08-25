# app/graph/exam_graph.py
#
# Architecture note: a Telegram conversation spans many separate webhook
# calls (one per message/button click), each of which must resume exactly
# where the previous one left off — not replay the whole flow. We use
# LangGraph's `interrupt()` primitive for this: a node calls `interrupt(...)`,
# execution pauses and the interrupt payload is returned to the caller, and
# the *next* webhook call resumes execution inside that same node (via
# `Command(resume=user_input)`) with `interrupt()` returning the user's input
# as if it had returned normally. The checkpointer persists graph state
# externally (Postgres in production), so a server restart does not lose an
# in-progress session (spec section 16/23).
#
# Each node calls `interrupt()` at most ONCE per execution and validates
# synchronously; on invalid input a conditional edge routes back to the SAME
# node (a fresh execution, fresh interrupt) rather than looping with multiple
# interrupt() calls inside one node body — the latter does not resume
# reliably against this LangGraph version's checkpointer.
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.base import BaseCheckpointSaver
from app.agents.state import ExamSessionState, SessionStep
from app.agents.session_manager import SessionManagerAgent
from app.agents.exam_resolver import ExamResolverAgent
from app.agents.curriculum_resolver import CurriculumResolverAgent
from app.agents.retrieval import RetrievalAgent
from app.agents.question_generator import QuestionGeneratorAgent
from app.agents.quiz_manager import QuizManagerAgent
from app.agents.evaluation import EvaluationAgent
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)


def _strip_prefix(user_input: Any, prefix: str) -> Optional[str]:
    if not isinstance(user_input, str) or not user_input.startswith(prefix):
        return None
    return user_input[len(prefix):]


def _allocate_by_weight(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Largest-remainder allocation of `total` items across `weights` (name ->
    share). Every entry with weight > 0 gets at least 1 item as long as there
    are enough items to go around; otherwise only the highest-weighted
    entries get one, so the total still matches exactly what was asked for."""
    entries = sorted(
        ((k, v) for k, v in weights.items() if v and v > 0),
        key=lambda kv: kv[1], reverse=True
    )
    if not entries or total <= 0:
        return {}

    if total <= len(entries):
        return {k: 1 for k, _ in entries[:total]}

    total_weight = sum(v for _, v in entries)
    raw = {k: total * v / total_weight for k, v in entries}
    alloc = {k: max(1, int(r)) for k, r in raw.items()}

    remainder = total - sum(alloc.values())
    by_fraction = sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)
    i = 0
    while remainder > 0:
        alloc[by_fraction[i % len(by_fraction)][0]] += 1
        remainder -= 1
        i += 1

    return alloc


class ExamGraph:
    """
    Main LangGraph orchestration for the exam preparation workflow.
    One compiled graph, one checkpointed "thread" per chat session
    (thread_id == ChatSession.id). Call `start()` for a brand new thread and
    `resume()` for every subsequent message on an existing thread.
    """

    def __init__(
        self,
        session_manager: SessionManagerAgent,
        exam_resolver: ExamResolverAgent,
        curriculum_resolver: CurriculumResolverAgent,
        retrieval_agent: RetrievalAgent,
        question_generator: QuestionGeneratorAgent,
        quiz_manager: QuizManagerAgent,
        evaluation_agent: EvaluationAgent,
        checkpointer: BaseCheckpointSaver
    ):
        self.session_manager = session_manager
        self.exam_resolver = exam_resolver
        self.curriculum_resolver = curriculum_resolver
        self.retrieval_agent = retrieval_agent
        self.question_generator = question_generator
        self.quiz_manager = quiz_manager
        self.evaluation_agent = evaluation_agent
        self.checkpointer = checkpointer
        self.graph = self._build_graph().compile(checkpointer=checkpointer)

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ExamSessionState)

        graph.add_node("resolve_exam", self._resolve_exam_node)
        graph.add_node("get_question_count", self._get_question_count_node)
        graph.add_node("resolve_chapter", self._resolve_chapter_node)
        graph.add_node("get_duration", self._get_duration_node)
        graph.add_node("generate_quiz", self._generate_quiz_node)
        graph.add_node("start_quiz", self._start_quiz_node)
        graph.add_node("handle_quiz", self._handle_quiz_node)
        graph.add_node("evaluate_quiz", self._evaluate_quiz_node)
        graph.add_node("show_results", self._show_results_node)

        graph.set_entry_point("resolve_exam")

        graph.add_conditional_edges(
            "resolve_exam", self._route_resolve_exam,
            {"retry": "resolve_exam", "next": "get_question_count"}
        )
        graph.add_conditional_edges(
            "get_question_count", self._route_question_count,
            {"retry": "get_question_count", "next": "resolve_chapter"}
        )
        graph.add_conditional_edges(
            "resolve_chapter", self._route_resolve_chapter,
            {"retry": "resolve_chapter", "next": "get_duration"}
        )
        graph.add_conditional_edges(
            "get_duration", self._route_duration,
            {"retry": "get_duration", "next": "generate_quiz"}
        )
        graph.add_conditional_edges(
            "generate_quiz", self._route_generate_quiz,
            {"retry": "resolve_chapter", "next": "start_quiz"}
        )
        graph.add_conditional_edges(
            "start_quiz", self._route_start_quiz,
            {"retry": "start_quiz", "next": "handle_quiz"}
        )
        graph.add_conditional_edges(
            "handle_quiz", self._route_handle_quiz,
            {"retry": "handle_quiz", "next": "evaluate_quiz"}
        )
        graph.add_edge("evaluate_quiz", "show_results")
        graph.add_edge("show_results", END)

        return graph

    # ---- run / resume -----------------------------------------------------

    def _config(self, thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def is_awaiting_input(self, thread_id: str) -> bool:
        """True if this thread is paused at an interrupt (mid-conversation); False if finished or new."""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        return bool(snapshot.next)


    async def start(self, state: ExamSessionState) -> Dict[str, Any]:
        """Begin a brand-new conversation thread"""
        await self.graph.ainvoke(state, config=self._config(state.session_id))
        return await self._state_or_prompt(state.session_id)

    async def resume(self, thread_id: str, user_input: Any) -> Dict[str, Any]:
        """Continue a paused conversation thread with the user's latest message/button value"""
        await self.graph.ainvoke(Command(resume=user_input), config=self._config(thread_id))
        return await self._state_or_prompt(thread_id)

    async def _state_or_prompt(self, thread_id: str) -> Dict[str, Any]:
        """
        After an ainvoke() call, inspect the checkpointed snapshot to determine
        whether the thread is now paused at an interrupt (mid-conversation) or
        has run to completion. Normalizes to
        {"state": <state-dict-or-None>, "prompt": <payload-or-None>}.
        """
        snapshot = await self.graph.aget_state(self._config(thread_id))
        if snapshot.next:
            for task in snapshot.tasks:
                if task.interrupts:
                    return {"state": None, "prompt": task.interrupts[0].value}
            return {"state": None, "prompt": {}}
        return {"state": snapshot.values, "prompt": None}

    async def get_prompt(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """The payload of the interrupt this thread is currently paused at, if any"""
        snapshot = await self.graph.aget_state(self._config(thread_id))
        for task in snapshot.tasks:
            if task.interrupts:
                return task.interrupts[0].value
        return None

    # ---- nodes: exam selection ---------------------------------------------

    async def _resolve_exam_node(self, state: ExamSessionState) -> ExamSessionState:
        if state.exam_id:
            return state

        user_input = interrupt({
            "step": SessionStep.SELECT_EXAM.value,
            "message": state.error or "Hello, let's study. Select exam from the main menu",
            "show_menu": True
        })
        state.error = None

        direct_id = _strip_prefix(user_input, "exam_")
        if direct_id:
            exam = await self.exam_resolver.exam_repo.get_by_id(direct_id)
            if exam and exam.is_active:
                state.exam_id, state.exam_name = str(exam.id), exam.name
                state.exam_confidence, state.exam_matched_method = 1.0, "button"
                return state

        exam_id, exam_name, confidence, method = await self.exam_resolver.resolve_exam(state, user_input)
        if exam_id and confidence >= settings.exam_match_confidence_threshold:
            state.exam_id, state.exam_name = exam_id, exam_name
            state.exam_confidence, state.exam_matched_method = confidence, method
        else:
            # Only checked after matching has already failed — never gates
            # whether matching is attempted, so it can't override a real
            # (if unsuccessful) exam-name attempt like a typo.
            if isinstance(user_input, str) and await self.exam_resolver.is_off_topic_message(user_input):
                state.error = "I'm a dedicated exam-prep bot, not built for general chat — please pick your exam from the menu below."
            else:
                state.error = "I couldn't confidently identify the exam. Please select one from the menu."
        return state

    def _route_resolve_exam(self, state: ExamSessionState) -> str:
        return "next" if state.exam_id else "retry"

    # ---- nodes: question count ---------------------------------------------

    async def _get_question_count_node(self, state: ExamSessionState) -> ExamSessionState:
        user_input = interrupt({
            "step": SessionStep.SELECT_QUESTION_COUNT.value,
            "message": state.error or "Choose number of questions to be generated"
        })
        state.error = None
        try:
            count = int(str(user_input).strip())
        except (ValueError, AttributeError):
            state.error = "Invalid input. Please enter a positive whole number."
            return state

        if count <= 0:
            state.error = "Invalid input. Please enter a positive number of questions."
        elif count > settings.max_questions_per_quiz:
            state.error = f"Maximum {settings.max_questions_per_quiz} questions allowed. Please enter a smaller number."
        else:
            state.question_count = count
        return state

    def _route_question_count(self, state: ExamSessionState) -> str:
        return "next" if state.question_count else "retry"

    # ---- nodes: chapter/topic -----------------------------------------------

    async def _resolve_chapter_node(self, state: ExamSessionState) -> ExamSessionState:
        # Coming back here from generate_quiz's "retry" edge after a generation failure
        state.chapter_id, state.chapter_name = None, None if not state.error else state.chapter_name

        options = await self.curriculum_resolver.get_options(state.exam_id)
        # A full curriculum can run into the hundreds of chapters/topics (this
        # exam has 222) — dumping all of them as buttons is unusable, and
        # Telegram messages/keyboards have their own size limits. Show a
        # capped browse list up front; after a failed free-text attempt, show
        # whatever's actually closest to what was typed (the closest thing
        # this UI has to a search box) instead of repeating the same generic
        # list, so anything outside that initial cap stays reachable.
        last_query = state.context.get("chapter_search_query")
        if state.error and last_query:
            display_options = self._closest_options(last_query, options, limit=10)
            message = f'{state.error}\n\nClosest matches to "{last_query}":'
        else:
            display_options = options[:30]
            message = state.error or (
                "Choose a chapter/topic for the test or select Overall "
                "(or type part of a name to search for it)."
            )

        user_input = interrupt({
            "step": SessionStep.SELECT_CHAPTER.value,
            "message": message,
            "options": display_options
        })
        state.error = None

        direct_id = _strip_prefix(user_input, "chapter_")
        if direct_id == "overall":
            state.chapter_id, state.chapter_name = "overall", "Overall"
            return state
        if direct_id:
            # Look up against the FULL option set, not just whatever page of
            # buttons this press came from, so every button always resolves.
            match = next((o for o in options if o["id"] == direct_id), None)
            if match:
                state.chapter_id, state.chapter_name = match["id"], match["name"]
                return state

        chapter_id, chapter_name, confidence, method = await self.curriculum_resolver.resolve_chapter(
            state.exam_id, user_input
        )
        if chapter_id == "overall":
            state.chapter_id, state.chapter_name = "overall", "Overall"
        elif chapter_id and confidence >= settings.chapter_match_confidence_threshold:
            state.chapter_id, state.chapter_name = chapter_id, chapter_name
        else:
            # Only checked after matching has already failed — see the
            # matching comment in _resolve_exam_node above.
            if isinstance(user_input, str) and await self.curriculum_resolver.is_off_topic_message(user_input):
                state.error = "I'm a dedicated exam-prep bot, not built for general chat — please pick a chapter/topic from the menu below."
                state.context["chapter_search_query"] = None
            else:
                state.error = "Couldn't find that chapter."
                state.context["chapter_search_query"] = user_input if isinstance(user_input, str) else None
        return state

    def _route_resolve_chapter(self, state: ExamSessionState) -> str:
        return "next" if state.chapter_name else "retry"

    @staticmethod
    def _closest_options(
        query: str, options: List[Dict[str, Any]], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Rank chapters/topics by textual similarity to whatever the user
        typed — a lightweight stand-in for a real search box, using only the
        stdlib (no new dependency, and these are short names so exact
        substring/character-overlap scoring is plenty)."""
        if not query:
            return options[:limit]
        import difflib
        q = query.lower()
        scored = [
            (difflib.SequenceMatcher(None, q, o["name"].lower()).ratio(), o)
            for o in options
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [o for _, o in scored[:limit]]

    # ---- nodes: duration -----------------------------------------------------

    async def _get_duration_node(self, state: ExamSessionState) -> ExamSessionState:
        user_input = interrupt({
            "step": SessionStep.SELECT_DURATION.value,
            "message": state.error or "Enter duration of test (in mins)."
        })
        state.error = None
        try:
            duration = int(str(user_input).strip())
        except (ValueError, AttributeError):
            state.error = "Invalid input. Please enter a valid duration in minutes."
            return state

        if duration <= 0:
            state.error = "Invalid input. Please enter a valid duration in minutes."
        elif duration > settings.max_quiz_duration_minutes:
            state.error = f"Maximum {settings.max_quiz_duration_minutes} minutes allowed. Please enter a smaller duration."
        else:
            state.duration_minutes = duration
        return state

    def _route_duration(self, state: ExamSessionState) -> str:
        return "next" if state.duration_minutes else "retry"

    # ---- nodes: quiz generation -----------------------------------------------

    async def _generate_quiz_node(self, state: ExamSessionState) -> ExamSessionState:
        # Real chapter_id is None for "overall"; the literal "overall" is just the UI sentinel
        chapter_filter = None if state.chapter_id == "overall" else state.chapter_id
        blueprint = state.context.get("blueprint", {})
        try:
            if chapter_filter is None:
                questions = await self._generate_overall_questions(state, blueprint)
            else:
                chunks = await self.retrieval_agent.retrieve_for_question_generation(
                    exam_id=state.exam_id,
                    chapter_id=chapter_filter,
                    query=state.chapter_name or "all topics"
                )
                questions = await self.question_generator.generate_questions(
                    exam_id=state.exam_id,
                    blueprint=blueprint,
                    retrieved_chunks=chunks,
                    chapter_name=state.chapter_name or "All Topics",
                    question_count=state.question_count,
                    difficulty_distribution=blueprint.get("difficulty", {"medium": 1.0})
                )

            if not questions:
                raise ValueError("no questions generated")

            quiz_id = await self.quiz_manager.create_quiz(state, questions)
            if not quiz_id:
                raise ValueError("quiz creation failed")

            state.quiz_id = quiz_id
            state.status = "quiz_ready"
            state.error = None
            state = await self.session_manager.transition_step(state, SessionStep.QUIZ_READY)
            return state

        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            state.error = "Sorry, I couldn't generate the test right now. Please choose a chapter/topic to try again."
            state.chapter_id, state.chapter_name = None, None
            return state

    def _route_generate_quiz(self, state: ExamSessionState) -> str:
        return "next" if state.quiz_id else "retry"

    async def _generate_overall_questions(
        self, state: ExamSessionState, blueprint: Dict[str, Any]
    ) -> list:
        """"Overall" must sample across the whole curriculum, not just whichever
        chapter happens to be nearest to a single generic query embedding — a
        plain vector search for "Overall" clusters in one semantic neighborhood
        (chunks about one chapter) and silently produces a single-topic quiz.
        Instead, split the requested question count across chapters using the
        blueprint's own chapter_distribution weights (the same weights the
        admin panel shows), retrieve chunks per chapter, and generate each
        chapter's share separately."""
        chapter_dist = blueprint.get("chapter_distribution") or {}
        allocation = _allocate_by_weight(state.question_count, chapter_dist)

        if not allocation:
            # No per-chapter breakdown on this blueprint (e.g. it predates
            # blueprint analysis, or only has subject-level data) — fall back
            # to the old single broad retrieval rather than failing outright.
            chunks = await self.retrieval_agent.retrieve_for_question_generation(
                exam_id=state.exam_id, chapter_id="", query=state.exam_name or "all topics"
            )
            return await self.question_generator.generate_questions(
                exam_id=state.exam_id,
                blueprint=blueprint,
                retrieved_chunks=chunks,
                chapter_name="Overall",
                question_count=state.question_count,
                difficulty_distribution=blueprint.get("difficulty", {"medium": 1.0})
            )

        all_questions = []
        for chapter_name, count in allocation.items():
            chunks = await self.retrieval_agent.retrieve_for_question_generation(
                exam_id=state.exam_id,
                chapter_id="",
                query=chapter_name,
                num_chunks=max(10, count * 3)
            )
            if not chunks:
                logger.warning(f"No chunks retrieved for chapter '{chapter_name}' in overall quiz; skipping it")
                continue
            chapter_questions = await self.question_generator.generate_questions(
                exam_id=state.exam_id,
                blueprint=blueprint,
                retrieved_chunks=chunks,
                chapter_name=chapter_name,
                question_count=count,
                difficulty_distribution=blueprint.get("difficulty", {"medium": 1.0})
            )
            all_questions.extend(chapter_questions)

        import random
        random.shuffle(all_questions)
        return all_questions

    # ---- nodes: quiz start / in-progress ---------------------------------------

    async def _start_quiz_node(self, state: ExamSessionState) -> ExamSessionState:
        user_input = interrupt({
            "step": SessionStep.QUIZ_READY.value,
            "message": state.error or "Your test is ready."
        })
        state.error = None
        if user_input == "start_quiz":
            await self.quiz_manager.start_quiz(state.quiz_id)
            state.start_time = datetime.utcnow()
            state.expiry_time = state.start_time + timedelta(minutes=state.duration_minutes)
            state.status = "in_progress"
            state = await self.session_manager.transition_step(state, SessionStep.QUIZ_IN_PROGRESS)
        else:
            # Previously silently re-showed the identical "Your test is
            # ready." message for ANY other input (a stray "hello", an old
            # session resurfacing days later, etc.) with zero feedback,
            # making it look like the bot was ignoring you / stuck forever.
            state.error = "Please press ▶ Start Quiz to begin, or send /start to begin a new session instead."
        return state

    def _route_start_quiz(self, state: ExamSessionState) -> str:
        return "next" if state.status == "in_progress" else "retry"

    async def _handle_quiz_node(self, state: ExamSessionState) -> ExamSessionState:
        # The scheduled expiry worker (app.workers.expiry) may have already force-submitted
        # this quiz while the user was inactive; check quiz status before prompting again.
        quiz = await self.quiz_manager.get_quiz(state.quiz_id)
        if quiz and quiz["status"] in ("submitted", "evaluated", "expired"):
            state.submitted_at = quiz.get("submitted_at") or datetime.utcnow()
            state.status = "submitted"
            return state

        if await self.quiz_manager.check_expiry(state):
            await self.quiz_manager.submit_quiz(state.quiz_id, state.answers)
            state.submitted_at = datetime.utcnow()
            state.status = "submitted"
            return state

        action = interrupt({
            "step": SessionStep.QUIZ_IN_PROGRESS.value,
            "questions": state.questions,
            "current_index": state.current_question_index,
            "answers": state.answers,
            "expiry_time": state.expiry_time.isoformat() if state.expiry_time else None
        })

        action = action if isinstance(action, dict) else {}
        action_type = action.get("type")

        if action_type == "answer":
            await self.quiz_manager.answer_question(state, action["question_index"], action["answer"])
        elif action_type == "nav":
            state.current_question_index = action["index"]
        elif action_type == "submit":
            await self.quiz_manager.submit_quiz(state.quiz_id, state.answers)
            state.submitted_at = datetime.utcnow()
            state.status = "submitted"
        return state

    def _route_handle_quiz(self, state: ExamSessionState) -> str:
        return "next" if state.status == "submitted" else "retry"

    # ---- nodes: evaluation / results -------------------------------------------

    async def _evaluate_quiz_node(self, state: ExamSessionState) -> ExamSessionState:
        if state.quiz_id:
            evaluation = await self.evaluation_agent.evaluate_quiz(state.quiz_id)
            state.evaluation = evaluation
            state.score = evaluation.get("score", 0)
            state = await self.session_manager.transition_step(state, SessionStep.RESULTS)
        return state

    async def _show_results_node(self, state: ExamSessionState) -> ExamSessionState:
        state.status = "completed"
        return state
