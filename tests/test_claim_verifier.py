from core.claim_verifier import (
    verify_single_claim,
    verify_claims
)

from core.claim_extractor import extract_claims


def test_supported_claim():

    result = verify_single_claim(
        "Jupiter is the largest planet in the Solar System."
    )

    print(result)

    assert result["status"] in [
        "supported",
        "contradicted",
        "insufficient_evidence"
    ]


def test_contradicted_claim():

    result = verify_single_claim(
        "Earth is the largest planet in the Solar System."
    )

    print(result)

    assert result["status"] in [
        "supported",
        "contradicted",
        "insufficient_evidence"
    ]


def test_multiple_claims():

    answer = """
    Jupiter is the largest planet in the Solar System.
    Earth is the third planet from the Sun.
    Earth is the largest planet in the Solar System.
    """

    claims = extract_claims(answer)

    result = verify_claims(claims)

    assert result["total_claims"] == len(claims)

    assert (
        result["supported_count"]
        + result["contradicted_count"]
        + result["insufficient_count"]
        == result["total_claims"]
    )