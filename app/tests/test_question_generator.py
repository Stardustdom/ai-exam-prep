# app/tests/test_question_generator.py
from app.agents.question_generator import QuestionGeneratorAgent


def make_agent():
    return QuestionGeneratorAgent(llm_service=None, embedding_service=None, quiz_repo=None)


def test_validate_question_rejects_missing_fields():
    agent = make_agent()
    assert not agent._validate_question({"question_text": "Q", "options": ["A", "B"]})  # no correct_answer


def test_validate_question_rejects_duplicate_options():
    agent = make_agent()
    q = {"question_text": "Q", "options": ["A", "A", "B", "C"], "correct_answer": "A"}
    assert not agent._validate_question(q)


def test_validate_question_rejects_correct_answer_not_in_options():
    agent = make_agent()
    q = {"question_text": "Q", "options": ["A", "B", "C"], "correct_answer": "Z"}
    assert not agent._validate_question(q)


def test_validate_question_accepts_well_formed_question():
    agent = make_agent()
    q = {"question_text": "What is 2+2?", "options": ["3", "4", "5", "6"], "correct_answer": "4"}
    assert agent._validate_question(q)


def test_fingerprint_stable_across_whitespace_variation():
    agent = make_agent()
    a = agent._generate_fingerprint({"question_text": "What   is velocity?"})
    b = agent._generate_fingerprint({"question_text": "what is velocity?  "})
    assert a == b


def test_fingerprint_differs_for_different_questions():
    agent = make_agent()
    a = agent._generate_fingerprint({"question_text": "What is velocity?"})
    b = agent._generate_fingerprint({"question_text": "What is acceleration?"})
    assert a != b


def test_distribution_sums_to_total_count():
    agent = make_agent()
    dist = agent._calculate_distribution(20, {"easy": 0.2, "medium": 0.5, "hard": 0.3})
    assert sum(dist.values()) == 20


def test_distribution_handles_uneven_rounding():
    agent = make_agent()
    dist = agent._calculate_distribution(10, {"easy": 0.33, "medium": 0.33, "hard": 0.34})
    assert sum(dist.values()) == 10
