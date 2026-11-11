from core.claim_extractor import extract_claims


def test_extract_claims():

    answer = """
    The Earth revolves around the Sun.
    Jupiter is the largest planet in our solar system.
    Thank you for asking!
    """

    claims = extract_claims(answer)

    assert len(claims) == 2

    assert claims[0]["claim_text"] == (
        "The Earth revolves around the Sun."
    )

    assert claims[1]["claim_text"] == (
        "Jupiter is the largest planet in our solar system."
    )


def test_empty_answer():

    claims = extract_claims("")

    assert claims == []


def test_claim_ids():

    answer = """
    Python is a programming language.
    Java supports object-oriented programming.
    """

    claims = extract_claims(answer)

    assert claims[0]["claim_id"] == 1
    assert claims[1]["claim_id"] == 2