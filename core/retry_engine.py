
from core.pipeline import run_pipeline


class RetryEngine:

    def __init__(self, generator, max_retries=3):
        self.generator = generator
        self.max_retries = max_retries

    # --------------------------------------------------------
    # HELPER: GET CLAIM TEXT
    # --------------------------------------------------------

    @staticmethod
    def get_claim_text(claim):
        return (
            claim.get("claim")
            or claim.get("claim_text")
            or ""
        ).strip()

    # --------------------------------------------------------
    # HELPER: GET VERIFICATION CLAIMS
    # --------------------------------------------------------

    @staticmethod
    def get_verified_claims(verification_result):
        return verification_result.get("claims", [])

    # --------------------------------------------------------
    # EXTRACT SUPPORTED CLAIMS
    # --------------------------------------------------------

    def get_supported_claims(self, verification_result):

        supported = []

        for claim in self.get_verified_claims(verification_result):

            if claim.get("status") == "supported":

                claim_text = self.get_claim_text(claim)

                if claim_text:
                    supported.append(claim_text)

        return supported

    # --------------------------------------------------------
    # CREATE FEEDBACK FOR UNSUPPORTED CLAIMS
    # --------------------------------------------------------

    def create_feedback(self, verification_result):

        feedback = []

        claims = self.get_verified_claims(verification_result)

        for claim in claims:

            if claim.get("status") != "supported":

                evidence = [
                    item.get("text", "")
                    for item in claim.get("evidence", [])
                    if item.get("text")
                ]

                feedback.append({
                    "claim": self.get_claim_text(claim),
                    "status": claim.get("status", "unknown"),
                    "reason": claim.get("reason", ""),
                    "evidence": evidence
                })

        return feedback

    # --------------------------------------------------------
    # FORMAT FEEDBACK
    # --------------------------------------------------------

    @staticmethod
    def format_feedback(feedback):

        sections = []

        for index, item in enumerate(feedback, start=1):

            evidence_text = (
                "\n".join(
                    f"- {evidence}"
                    for evidence in item["evidence"]
                )
                if item["evidence"]
                else "- No relevant evidence retrieved."
            )

            section = f"""
Claim {index}: {item['claim']}

Status: {item['status']}

Reason:
{item['reason']}

Retrieved evidence:
{evidence_text}
"""

            sections.append(section)

        return "\n".join(sections)

    # --------------------------------------------------------
    # CREATE CLAIM-LEVEL CORRECTION PROMPT
    # --------------------------------------------------------

    def create_correction_prompt(
        self,
        initial_prompt,
        supported_claims,
        feedback
    ):

        supported_text = (
            "\n".join(
                f"- {claim}"
                for claim in supported_claims
            )
            if supported_claims
            else "No claims have been verified yet."
        )

        feedback_text = self.format_feedback(feedback)

        return f"""
Original user question:
{initial_prompt}

CLAIMS ALREADY VERIFIED BY HALLUGUARD:

{supported_text}

THE FOLLOWING CLAIMS NEED CORRECTION:

{feedback_text}

YOUR TASK:

Correct ONLY the claims listed under "claims need correction".

STRICT INSTRUCTIONS:

1. Do not rewrite or repeat the already verified claims.
2. Do not invent facts or fabricate supporting evidence.
3. Use the retrieved evidence when it supports a correction.
4. If a claim is contradicted, provide a corrected factual statement
   only when the evidence supports that correction.
5. If the available evidence is insufficient, do not guess.
6. Do not write explanations such as "the evidence is insufficient".
7. Do not return the original unsupported claim.
8. Return only the corrected factual statement(s).
9. If none of the claims can be corrected using the evidence,
   return exactly:

NO_SUPPORTED_CORRECTION

Do not include introductions, conclusions, or unrelated facts.
"""

    # --------------------------------------------------------
    # REMOVE DUPLICATE CLAIMS
    # --------------------------------------------------------

    @staticmethod
    def remove_duplicates(claims):

        unique_claims = []
        seen = set()

        for claim in claims:

            normalized = " ".join(claim.lower().split())

            if normalized and normalized not in seen:

                seen.add(normalized)
                unique_claims.append(claim)

        return unique_claims

    # --------------------------------------------------------
    # BUILD FINAL ANSWER
    # --------------------------------------------------------

    def build_final_answer(self, supported_claims):

        supported_claims = self.remove_duplicates(
            supported_claims
        )

        formatted_claims = []

        for claim in supported_claims:

            claim = claim.strip()

            # Ensure every claim has a sentence boundary
            if claim and claim[-1] not in ".!?":
                claim += "."

            formatted_claims.append(claim)

        return "\n".join(
            f"{index}. {claim}"
            for index, claim in enumerate(
                formatted_claims,
                start=1
            )
        )


    # --------------------------------------------------------
    # RUN GENERATION AND VERIFICATION
    # --------------------------------------------------------

    def run(self, initial_prompt):

        print("\n========== HALLUGUARD STARTED ==========")

        history = []

        # Initial generation
        current_answer = self.generator.generate(initial_prompt)

        # Verify initial answer
        print("\n========== ATTEMPT 1: INITIAL ANSWER ==========")

        result = run_pipeline(current_answer)

        history.append({
            "attempt": 1,
            "answer": current_answer,
            "decision": result["review"]["decision"],
            "review_score": result["review"]["review_score"]
        })

        # Preserve supported claims
        supported_claims = self.get_supported_claims(
            result["verification"]
        )

        # Get claims requiring correction
        pending_feedback = self.create_feedback(
            result["verification"]
        )

        # ----------------------------------------------------
        # CLAIM-LEVEL RETRY LOOP
        # ----------------------------------------------------

        for retry_number in range(1, self.max_retries + 1):

            # Stop if every claim is supported
            if not pending_feedback:
                break

            print(
                f"\n========== CORRECTION ATTEMPT "
                f"{retry_number} =========="
            )

            correction_prompt = self.create_correction_prompt(
                initial_prompt,
                supported_claims,
                pending_feedback
            )

            print("\nSending only unresolved claims to generator...")

            correction_answer = self.generator.generate(
                correction_prompt
            )

            # Handle the no-correction response
            if correction_answer.strip() == "NO_SUPPORTED_CORRECTION":

                print("\nGenerator could not provide a supported correction.")

                history.append({
                    "attempt": retry_number + 1,
                    "answer": correction_answer,
                    "decision": "no_supported_correction",
                    "review_score": None
                })

                break

            # Verify proposed corrections independently
            correction_result = run_pipeline(correction_answer)

            correction_verification = correction_result[
                "verification"
            ]

            correction_review = correction_result["review"]

            history.append({
                "attempt": retry_number + 1,
                "answer": correction_answer,
                "decision": correction_review["decision"],
                "review_score": correction_review["review_score"]
            })

            # Accept only supported corrections
            newly_supported = self.get_supported_claims(
                correction_verification
            )

            supported_claims.extend(newly_supported)

            supported_claims = self.remove_duplicates(
                supported_claims
            )

            # Keep unresolved corrections for the next retry
            new_feedback = self.create_feedback(
                correction_verification
            )

            if newly_supported:

                print(
                    f"\nNew supported corrections: "
                    f"{len(newly_supported)}"
                )

            # Update pending claims
            if new_feedback:
                pending_feedback = new_feedback
            else:
                pending_feedback = []

            print(
                f"Total supported claims: "
                f"{len(supported_claims)}"
            )

            print(
                f"Remaining unresolved claims: "
                f"{len(pending_feedback)}"
            )

        # ----------------------------------------------------
        # BUILD FINAL VERIFIED ANSWER
        # ----------------------------------------------------

        final_answer = self.build_final_answer(
            supported_claims
        )

        # Verify the combined final answer
        if final_answer:

            print("\n========== FINAL ANSWER VERIFICATION ==========")

            final_verification = run_pipeline(final_answer)

            final_review = final_verification["review"]

            # Only fully approve if no claims remain unresolved
            if not pending_feedback and final_review["decision"] == "approved":
                decision = "approved"

            elif supported_claims:
                decision = "partially_approved"

            else:
                decision = "rejected"

        else:

            final_verification = result
            decision = "rejected"

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print("\n========== HALLUGUARD RESULT ==========")

        print("Decision:", decision)

        print("Supported claims:", len(supported_claims))

        print("Unresolved claims:", len(pending_feedback))

        print("\nFinal Answer:")
        print(final_answer if final_answer else "No verified claims available.")

        if pending_feedback:

            print("\nOmitted claims:")

            for item in pending_feedback:
                print("-", item["claim"], "|", item["status"])

        return {
            "final_answer": final_answer,
            "decision": decision,
            "attempts": len(history),
            "history": history,
            "supported_claims": supported_claims,
            "omitted_claims": pending_feedback,
            "verification": final_verification
        }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    from integrations.generators.mock_generator import MockGenerator

    generator = MockGenerator()

    retry_engine = RetryEngine(
        generator=generator,
        max_retries=3
    )

    result = retry_engine.run(
        "Describe some important facts about the Solar System."
    )

    print("\n========== FINAL RETRY RESULT ==========")

    print("Decision:", result["decision"])

    print("Attempts:", result["attempts"])

    print("\nFinal Answer:")
    print(result["final_answer"])

    print("\nOmitted Claims:")
    print(result["omitted_claims"])

    print("\nRetry History:")

    for item in result["history"]:
        print(item)
