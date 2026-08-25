# app/tests/test_matching.py
#
# Pure-function coverage for the normalization/similarity helpers behind
# exam and chapter/topic semantic resolution (spec sections 10, 12).
import pytest
from app.agents.exam_resolver import ExamResolverAgent
from app.agents.curriculum_resolver import CurriculumResolverAgent, _OVERALL_ALIASES


@pytest.fixture
def exam_resolver():
    return ExamResolverAgent(exam_repo=None, semantic_cache_repo=None, embedding_service=None, semantic_cache_service=None)


@pytest.fixture
def curriculum_resolver():
    return CurriculumResolverAgent(chapter_repo=None, topic_repo=None, embedding_service=None, semantic_cache_service=None)


def test_exam_normalize_lowercases_and_strips_stopwords(exam_resolver):
    assert exam_resolver._normalize_input("Joint   Entrance Examination") == "joint entrance examination"
    assert exam_resolver._normalize_input("The JEE Exam") == "jee"


def test_curriculum_normalize_strips_punctuation(curriculum_resolver):
    assert curriculum_resolver._normalize("Laws Of Motion!") == "laws of motion"
    assert curriculum_resolver._normalize("  kinematics  ") == "kinematics"


@pytest.mark.parametrize("phrase", ["overall", "Overall", "ALL", "everything", "full syllabus"])
def test_overall_aliases_recognized(phrase):
    assert phrase.lower().strip() in _OVERALL_ALIASES


def test_cosine_similarity_identical_vectors_is_one(curriculum_resolver):
    v = [1.0, 2.0, 3.0]
    assert curriculum_resolver._cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero(curriculum_resolver):
    assert curriculum_resolver._cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_safe(curriculum_resolver):
    assert curriculum_resolver._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
