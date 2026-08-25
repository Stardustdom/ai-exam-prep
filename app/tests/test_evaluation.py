# app/tests/test_evaluation.py
#
# Deterministic scoring, including blueprint-driven negative marking
# (spec section 25) — never delegated to an LLM.
import pytest
from datetime import datetime, timedelta
from app.agents.evaluation import EvaluationAgent


class FakeQuiz:
    def __init__(self, questions, user_answers, exam_id="exam-1"):
        self.id = "quiz-1"
        self.exam_id = exam_id
        self.chapter_id = None
        self.generated_questions = questions
        self.user_answers = user_answers
        self.start_time = datetime(2026, 1, 1, 10, 0, 0)
        self.submitted_at = datetime(2026, 1, 1, 10, 20, 0)


class FakeQuizRepo:
    def __init__(self, quiz):
        self.quiz = quiz
        self.updated_with = None

    async def get_by_id(self, quiz_id):
        return self.quiz

    async def update(self, quiz_id, data):
        self.updated_with = data


class FakeBlueprint:
    def __init__(self, blueprint_data):
        self.blueprint_data = blueprint_data


class FakeBlueprintRepo:
    def __init__(self, blueprint_data):
        self._blueprint = FakeBlueprint(blueprint_data) if blueprint_data else None

    async def get_active_by_exam(self, exam_id):
        return self._blueprint


class FakeLLM:
    async def generate_json(self, prompt, system=""):
        return {"strong_areas": [], "weak_areas": [], "overall_assessment": "", "recommendations": []}


QUESTIONS = [
    {"question_text": "Q1", "correct_answer": "A", "difficulty": "easy"},
    {"question_text": "Q2", "correct_answer": "B", "difficulty": "medium"},
    {"question_text": "Q3", "correct_answer": "C", "difficulty": "hard"},
    {"question_text": "Q4", "correct_answer": "D", "difficulty": "medium"},
]


@pytest.mark.asyncio
async def test_simple_scoring_no_blueprint():
    # 2 correct (idx 0, 1), 1 wrong (idx 2 answered "X"), 1 unanswered (idx 3)
    answers = [
        {"question_index": 0, "answer": "A"},
        {"question_index": 1, "answer": "B"},
        {"question_index": 2, "answer": "X"},
    ]
    quiz = FakeQuiz(QUESTIONS, answers)
    agent = EvaluationAgent(FakeQuizRepo(quiz), FakeLLM(), retrieval_service=None, blueprint_repo=None)

    result = await agent.evaluate_quiz("quiz-1")

    assert result["correct"] == 2
    assert result["incorrect"] == 1
    assert result["unanswered"] == 1
    assert result["attempted"] == 3
    assert result["score"] == 2  # no marking scheme configured -> 1 point per correct, no penalty
    assert result["accuracy"] == pytest.approx(2 / 3 * 100)
    assert result["time_taken_seconds"] == 1200


@pytest.mark.asyncio
async def test_negative_marking_applied_from_blueprint():
    answers = [
        {"question_index": 0, "answer": "A"},   # correct
        {"question_index": 1, "answer": "Z"},   # wrong
        {"question_index": 2, "answer": "Z"},   # wrong
    ]
    quiz = FakeQuiz(QUESTIONS, answers)
    blueprint_data = {
        "marking_scheme": {
            "average_marks_per_question": 4,
            "negative_marking": True,
            "average_negative_marks": 1,
        }
    }
    agent = EvaluationAgent(
        FakeQuizRepo(quiz), FakeLLM(), retrieval_service=None,
        blueprint_repo=FakeBlueprintRepo(blueprint_data)
    )

    result = await agent.evaluate_quiz("quiz-1")

    # 1 correct * 4 - 2 incorrect * 1 = 2
    assert result["score"] == 2
    assert result["correct"] == 1
    assert result["incorrect"] == 2


@pytest.mark.asyncio
async def test_unanswered_never_penalized():
    answers = []  # nothing answered
    quiz = FakeQuiz(QUESTIONS, answers)
    blueprint_data = {"marking_scheme": {"average_marks_per_question": 4, "negative_marking": True, "average_negative_marks": 1}}
    agent = EvaluationAgent(
        FakeQuizRepo(quiz), FakeLLM(), retrieval_service=None,
        blueprint_repo=FakeBlueprintRepo(blueprint_data)
    )

    result = await agent.evaluate_quiz("quiz-1")

    assert result["unanswered"] == 4
    assert result["score"] == 0
    assert result["accuracy"] == 0
