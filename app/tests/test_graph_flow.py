# app/tests/test_graph_flow.py
#
# Exercises the LangGraph interrupt/resume state machine end-to-end with fake
# agents (no DB/LLM), covering: multi-turn progression, invalid-input retry
# staying on the same step (spec section 40), and full quiz completion.
import pytest
from langgraph.checkpoint.memory import MemorySaver
from app.graph.exam_graph import ExamGraph
from app.agents.state import ExamSessionState, SessionStep


class FakeExam:
    def __init__(self, id, name, is_active=True):
        self.id, self.name, self.is_active = id, name, is_active


class FakeExamRepo:
    async def get_by_id(self, exam_id):
        return FakeExam(exam_id, "Joint Entrance Examination") if exam_id == "exam-1" else None


class FakeExamResolver:
    def __init__(self):
        self.exam_repo = FakeExamRepo()

    async def resolve_exam(self, state, user_input):
        if user_input and "jee" in user_input.lower():
            return "exam-1", "Joint Entrance Examination", 0.95, "semantic"
        return None, None, 0.0, "none"


class FakeCurriculumResolver:
    async def get_options(self, exam_id):
        return [{"id": "ch-1", "name": "Kinematics", "type": "chapter"}]

    async def resolve_chapter(self, exam_id, user_input):
        if user_input and "kinemat" in user_input.lower():
            return "ch-1", "Kinematics", 0.9, "semantic"
        return None, None, 0.0, "none"


class FakeSessionManager:
    async def transition_step(self, state, new_step):
        state.previous_step = state.current_step
        state.current_step = new_step
        return state


class FakeRetrieval:
    async def retrieve_for_question_generation(self, **kwargs):
        return [{"content": "Kinematics is the study of motion."}]


class FakeQuestionGen:
    def __init__(self, questions=None):
        self._questions = questions or [{
            "question_text": "What is velocity?",
            "options": ["Speed", "Rate of change of displacement", "Force", "Mass"],
            "correct_answer": "Rate of change of displacement",
            "difficulty": "easy"
        }]

    async def generate_questions(self, **kwargs):
        return self._questions


class FakeQuizManager:
    def __init__(self):
        self.quizzes = {}

    async def create_quiz(self, state, questions):
        qid = "quiz-1"
        self.quizzes[qid] = {"status": "generated", "questions": questions, "answers": {}, "submitted_at": None}
        state.questions = questions
        return qid

    async def start_quiz(self, quiz_id):
        self.quizzes[quiz_id]["status"] = "started"

    async def get_quiz(self, quiz_id):
        return self.quizzes.get(quiz_id)

    async def check_expiry(self, state):
        return False

    async def get_next_question(self, state):
        return state.questions[0] if state.questions else None

    async def answer_question(self, state, idx, answer):
        state.answers[idx] = answer

    async def submit_quiz(self, quiz_id, answers):
        self.quizzes[quiz_id]["status"] = "submitted"
        self.quizzes[quiz_id]["answers"] = answers


class FakeEvaluation:
    async def evaluate_quiz(self, quiz_id):
        return {"total_questions": 1, "correct": 1, "score": 1, "accuracy": 100.0}


def build_graph():
    return ExamGraph(
        session_manager=FakeSessionManager(),
        exam_resolver=FakeExamResolver(),
        curriculum_resolver=FakeCurriculumResolver(),
        retrieval_agent=FakeRetrieval(),
        question_generator=FakeQuestionGen(),
        quiz_manager=FakeQuizManager(),
        evaluation_agent=FakeEvaluation(),
        checkpointer=MemorySaver()
    )


@pytest.mark.asyncio
async def test_start_pauses_at_exam_selection():
    graph = build_graph()
    state = ExamSessionState(session_id="t1", telegram_chat_id="1", current_step=SessionStep.START)
    r = await graph.start(state)
    assert r["prompt"]["step"] == "select_exam"


@pytest.mark.asyncio
async def test_invalid_question_count_stays_on_same_step():
    graph = build_graph()
    state = ExamSessionState(session_id="t2", telegram_chat_id="1", current_step=SessionStep.START)
    await graph.start(state)
    await graph.resume("t2", "I want jee")

    r = await graph.resume("t2", "not a number")
    assert r["prompt"]["step"] == "select_question_count"
    assert "positive whole number" in r["prompt"]["message"]

    r = await graph.resume("t2", "-5")
    assert r["prompt"]["step"] == "select_question_count"

    r = await graph.resume("t2", "5")
    assert r["prompt"]["step"] == "select_chapter"


@pytest.mark.asyncio
async def test_unresolvable_exam_reprompts_with_menu():
    graph = build_graph()
    state = ExamSessionState(session_id="t3", telegram_chat_id="1", current_step=SessionStep.START)
    await graph.start(state)

    r = await graph.resume("t3", "some gibberish exam name")
    assert r["prompt"]["step"] == "select_exam"
    assert r["prompt"]["show_menu"] is True


@pytest.mark.asyncio
async def test_full_quiz_completes_and_thread_ends():
    graph = build_graph()
    state = ExamSessionState(session_id="t4", telegram_chat_id="1", current_step=SessionStep.START)
    await graph.start(state)
    await graph.resume("t4", "jee")
    await graph.resume("t4", "5")
    await graph.resume("t4", "kinematics")
    r = await graph.resume("t4", "30")
    assert r["prompt"]["step"] == "quiz_ready"

    r = await graph.resume("t4", "start_quiz")
    assert r["prompt"]["step"] == "quiz_in_progress"

    r = await graph.resume("t4", {"type": "answer", "question_index": 0, "answer": "Rate of change of displacement"})
    assert r["prompt"]["step"] == "quiz_in_progress"  # still awaiting explicit submit

    r = await graph.resume("t4", {"type": "submit"})
    assert r["prompt"] is None
    assert r["state"]["evaluation"]["correct"] == 1
    assert not await graph.is_awaiting_input("t4")


@pytest.mark.asyncio
async def test_button_selection_bypasses_semantic_matching():
    """A menu button click (exam_<id>) should resolve deterministically even
    for input the semantic matcher would reject."""
    graph = build_graph()
    state = ExamSessionState(session_id="t5", telegram_chat_id="1", current_step=SessionStep.START)
    await graph.start(state)
    r = await graph.resume("t5", "exam_exam-1")
    assert r["prompt"]["step"] == "select_question_count"
