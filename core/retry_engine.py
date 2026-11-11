
from core.pipeline import run_pipeline


class RetryEngine:

    def __init__(self, generator, max_retries=3):

        self.generator = generator
        self.max_retries = max_retries

    # --------------------------------------------------------
    # CREATE FEEDBACK FOR THE GENERATOR
    # --------------------------------------------------------

    def create_feedback(self, verification_result):

        feedback = []

        claims = verification_result.get("claims", [])

        for claim in claims:

            if claim["status"] != "supported":

                feedback.append({

                    "claim": claim["claim"],

                    "status": claim["status"],

                    "reason": claim["reason"],

                    "evidence": [
                        item["text"]
                        for item in claim.get("evidence", [])
                    ]

                })

        return feedback

    # --------------------------------------------------------
    # RUN GENERATION AND VERIFICATION
    # --------------------------------------------------------

    def run(self, initial_prompt):

        print("\n========== RETRY ENGINE STARTED ==========")

        current_answer = self.generator.generate(initial_prompt)

        history = []

        for attempt in range(self.max_retries + 1):

            print(f"\n========== ATTEMPT {attempt + 1} ==========")

            # Verify current answer
            result = run_pipeline(current_answer)

            review = result["review"]

            history.append({

                "attempt": attempt + 1,

                "answer": current_answer,

                "decision": review["decision"],

                "review_score": review["review_score"]

            })

            # If approved, stop
            if review["decision"] == "approved":

                print("\nANSWER APPROVED!")

                return {

                    "final_answer": current_answer,

                    "decision": "approved",

                    "attempts": attempt + 1,

                    "history": history,

                    "verification": result

                }

            # If all attempts are exhausted, stop
            if attempt == self.max_retries:

                break

            # Generate feedback
            feedback = self.create_feedback(
                result["verification"]
            )

            feedback_prompt = f"""
Original user prompt:
{initial_prompt}

Previous AI answer:
{current_answer}

The following claims were rejected by HalluGuard:

{feedback}

Please regenerate the complete answer.

Instructions:
1. Correct the rejected claims.
2. Use the supplied evidence.
3. Do not repeat unsupported claims.
4. If the evidence is insufficient, explicitly say so.
5. Preserve the original user's question.
"""

            print("\nAnswer rejected.")

            print("Sending feedback to generator...")

            # Request a revised answer
            current_answer = self.generator.generate(
                feedback_prompt
            )

        # ----------------------------------------------------
        # MAXIMUM RETRIES REACHED
        # ----------------------------------------------------

        print("\nMAXIMUM RETRIES REACHED.")

        return {

            "final_answer": current_answer,

            "decision": "rejected",

            "attempts": self.max_retries + 1,

            "history": history,

            "verification": result

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

    print("\nRetry History:")

    for item in result["history"]:

        print(item)
