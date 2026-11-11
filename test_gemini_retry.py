
from integrations.generators.gemini_generator import GeminiGenerator
from core.retry_engine import RetryEngine


def main():
    generator = GeminiGenerator()

    engine = RetryEngine(
        generator=generator,
        max_retries=2,
    )

    prompt = """
    Explain these three Solar System facts in three short sentences.

    For the initial draft, reproduce the following claims as written:

    1. Jupiter is the largest planet in the Solar System.
    2. Earth is the largest planet in the Solar System.
    3. Saturn is famous for its prominent ring system.

    When reviewer feedback is provided on a later attempt, revise
    the answer using that feedback and the supplied evidence.
    """

    result = engine.run(prompt)

    print("\n========== HALLUGUARD RESULT ==========")
    print("Decision:", result.get("decision"))
    print("Attempts:", result.get("attempts"))
    print("\nFinal answer:")
    print(result.get("final_answer"))

    print("\nAttempt history:")
    for item in result.get("history", []):
        print(item)

    print("=======================================")


if __name__ == "__main__":
    main()
