
# ============================================================
# HALLUGUARD AI - STRICT REVIEWER
# ============================================================

class StrictReviewer:

    def __init__(self, approval_threshold=0.95):

        self.approval_threshold = approval_threshold

    # --------------------------------------------------------
    # Review the complete AI-generated answer
    # --------------------------------------------------------

    def review_answer(self, verification_result):

        claims = verification_result.get("claims", [])

        total_claims = len(claims)

        if total_claims == 0:

            return {
                "decision": "rejected",
                "review_score": 0.0,
                "reason": "No verifiable factual claims were extracted.",
                "total_claims": 0,
                "approved_claims": 0,
                "rejected_claims": 0
            }

        supported = sum(
            claim["status"] == "supported"
            for claim in claims
        )

        contradicted = sum(
            claim["status"] == "contradicted"
            for claim in claims
        )

        insufficient = sum(
            claim["status"] == "insufficient_evidence"
            for claim in claims
        )

        # ----------------------------------------------------
        # Calculate review score
        # ----------------------------------------------------

        review_score = supported / total_claims

        # ----------------------------------------------------
        # Make approval decision
        # ----------------------------------------------------

        if (
            review_score >= self.approval_threshold
            and contradicted == 0
            and insufficient == 0
        ):

            decision = "approved"

            reason = (
                "All extracted factual claims are supported "
                "by the available evidence."
            )

        else:

            decision = "rejected"

            reasons = []

            if contradicted > 0:
                reasons.append(
                    f"{contradicted} contradicted claim(s)"
                )

            if insufficient > 0:
                reasons.append(
                    f"{insufficient} claim(s) lack sufficient evidence"
                )

            if review_score < self.approval_threshold:
                reasons.append(
                    "Review score is below the approval threshold"
                )

            reason = "; ".join(reasons)

        return {

            "decision": decision,

            "review_score": round(review_score, 4),

            "approval_threshold": self.approval_threshold,

            "reason": reason,

            "total_claims": total_claims,

            "approved_claims": supported,

            "rejected_claims": contradicted + insufficient,

            "supported_count": supported,

            "contradicted_count": contradicted,

            "insufficient_count": insufficient

        }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    reviewer = StrictReviewer(
        approval_threshold=0.95
    )

    sample_result = {

        "claims": [

            {"status": "supported"},

            {"status": "supported"},

            {"status": "supported"},

            {"status": "supported"}

        ]

    }

    result = reviewer.review_answer(sample_result)

    print("\n========== STRICT REVIEWER ==========")

    for key, value in result.items():

        print(f"{key}: {value}")
