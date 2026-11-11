
# ============================================================
# HALLUGUARD AI - MAIN VERIFICATION PIPELINE
# ============================================================

from core.claim_extractor import extract_claims
from core.claim_verifier import verify_claims
from core.strict_reviewer import StrictReviewer


# ============================================================
# INITIALIZE STRICT REVIEWER
# ============================================================

reviewer = StrictReviewer(
    approval_threshold=0.95
)


# ============================================================
# MAIN PIPELINE FUNCTION
# ============================================================

def run_pipeline(answer):

    # --------------------------------------------------------
    # STEP 1: Validate input
    # --------------------------------------------------------

    if not isinstance(answer, str) or not answer.strip():

        return {
            "original_answer": answer,
            "extracted_claims": [],
            "verification": {},
            "review": {
                "decision": "rejected",
                "review_score": 0.0,
                "reason": "Empty or invalid answer."
            }
        }

    # --------------------------------------------------------
    # STEP 2: Extract factual claims
    # --------------------------------------------------------

    print("\n========== STEP 1: CLAIM EXTRACTION ==========")

    claims = extract_claims(answer)

    print("Total claims extracted:", len(claims))

    for claim in claims:

        print(
            f"{claim['claim_id']}. {claim['claim_text']}"
        )

    # --------------------------------------------------------
    # STEP 3: Verify extracted claims
    # --------------------------------------------------------

    print("\n========== STEP 2: CLAIM VERIFICATION ==========")

    verification_result = verify_claims(claims)

    print(
        "Supported:",
        verification_result["supported_count"]
    )

    print(
        "Contradicted:",
        verification_result["contradicted_count"]
    )

    print(
        "Insufficient:",
        verification_result["insufficient_count"]
    )

    # --------------------------------------------------------
    # STEP 4: Strict Reviewer
    # --------------------------------------------------------

    print("\n========== STEP 3: STRICT REVIEWER ==========")

    review_result = reviewer.review_answer(
        verification_result
    )

    print("Decision:", review_result["decision"])

    print("Review score:", review_result["review_score"])

    print("Reason:", review_result["reason"])

    # --------------------------------------------------------
    # STEP 5: Prepare final output
    # --------------------------------------------------------

    final_result = {

        "original_answer": answer,

        "extracted_claims": claims,

        "verification": verification_result,

        "review": review_result

    }

    return final_result


# ============================================================
# MANUAL TESTING
# ============================================================

if __name__ == "__main__":

    sample_answer = """
    Jupiter is the largest planet in the Solar System.

    Earth is the third planet from the Sun.

    Earth is the largest planet in the Solar System.

    Saturn has a prominent ring system.
    """

    result = run_pipeline(sample_answer)

    print("\n========== FINAL HALLUGUARD RESULT ==========")

    print("\nOriginal Answer:")
    print(result["original_answer"])

    print("\nReview Decision:")
    print(result["review"]["decision"])

    print("\nReview Score:")
    print(result["review"]["review_score"])

    print("\nReason:")
    print(result["review"]["reason"])

    print("\n========== CLAIM DETAILS ==========")

    for claim in result["verification"].get("claims", []):

        print("\nClaim:", claim["claim"])

        print("Status:", claim["status"])

        print("Reason:", claim["reason"])
