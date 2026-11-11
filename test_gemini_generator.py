
from integrations.generators.gemini_generator import GeminiGenerator


def main():
    generator = GeminiGenerator()

    prompt = """
    Answer using these known facts about the Solar System:

    - Jupiter is the largest planet.
    - Earth is the third planet from the Sun.
    - Saturn has a prominent ring system.

    Write three short factual sentences.
    """

    answer = generator.generate(prompt)

    print("\n========== GEMINI RESPONSE ==========")
    print(answer)
    print("=====================================")


if __name__ == "__main__":
    main()
