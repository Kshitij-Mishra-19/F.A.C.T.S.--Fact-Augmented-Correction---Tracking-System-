
import spacy
import re


# Load the English language model
nlp = spacy.load("en_core_web_sm")


def clean_text(text):
    """
    Remove unnecessary formatting and list numbering.
    """

    # Remove fenced code blocks
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)

    # Remove inline code formatting
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove Markdown headings
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)

    # Remove bullet points and numbered list prefixes
    text = re.sub(
        r"(?m)^\s*(?:[-*+]|\d+[.)])\s+",
        "",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_claim(claim):
    """
    Remove leftover numbering and formatting from a claim.
    """

    claim = claim.strip()

    # Remove repeated numbering, e.g. '2. 2. Earth is...'
    while True:
        cleaned = re.sub(
            r"^\s*(?:\d+[.)]\s*|[-*+]\s*)",
            "",
            claim
        )

        if cleaned == claim:
            break

        claim = cleaned

    return claim.strip()


def split_compound_claim(sentence):
    """
    Split simple coordinated predicates into independent claims.

    Example:
    Earth is the third planet from the Sun and the largest
    of the terrestrial planets.

    Becomes:
    Earth is the third planet from the Sun.
    Earth is the largest of the terrestrial planets.

    This is a rule-based decomposition, not a complete
    natural-language claim decomposition system.
    """

    sentence = normalize_claim(sentence)

    pattern = (
        r"^(?P<subject>.+?)\s+"
        r"(?P<verb>is|are|was|were)\s+"
        r"(?P<first>.+?)\s+"
        r"(?P<conjunction>and|but)\s+"
        r"(?:(?P<second_verb>is|are|was|were)\s+)?"
        r"(?P<second>.+)$"
    )

    match = re.match(pattern, sentence, flags=re.IGNORECASE)

    if not match:
        return [sentence]

    subject = match.group("subject").strip()
    verb = match.group("verb").strip()
    first = match.group("first").strip()
    conjunction = match.group("conjunction").lower()
    second = match.group("second").strip()
    second_verb = match.group("second_verb")

    # Avoid splitting fragments that do not look like
    # independent predicates.
    if not first or not second:
        return [sentence]

    # Reconstruct the shared subject and verb.
    first_claim = f"{subject} {verb} {first}"

    second_auxiliary = second_verb or verb
    second_claim = f"{subject} {second_auxiliary} {second}"

    # Preserve the original conjunction's meaning.
    # 'but' indicates contrast, but both clauses remain claims.
    return [
        normalize_claim(first_claim),
        normalize_claim(second_claim)
    ]


def is_factual_candidate(sentence):
    """
    Basic filtering to identify sentences that may contain
    factual information.

    This is a candidate filter, NOT a truth detector.
    """

    sentence = sentence.strip()

    if len(sentence.split()) < 3:
        return False

    # Ignore questions
    if sentence.endswith("?"):
        return False

    ignored_phrases = [
        "hello",
        "hi there",
        "thank you",
        "you're welcome",
        "i hope this helps",
        "let me know",
        "great question"
    ]

    lower_sentence = sentence.lower()

    if any(
        lower_sentence.startswith(phrase)
        for phrase in ignored_phrases
    ):
        return False

    return True


def extract_claims(answer):
    """
    Extract factual candidate statements from an AI-generated answer.

    Returns a list of structured claim objects.
    """

    if not isinstance(answer, str) or not answer.strip():
        return []

    cleaned_answer = clean_text(answer)

    doc = nlp(cleaned_answer)

    claims = []
    seen_claims = set()

    for sentence in doc.sents:

        sentence_text = normalize_claim(sentence.text)

        if not is_factual_candidate(sentence_text):
            continue

        # Decompose simple compound statements
        individual_claims = split_compound_claim(sentence_text)

        for claim_text in individual_claims:

            claim_text = normalize_claim(claim_text)

            if not is_factual_candidate(claim_text):
                continue

            # Avoid duplicate claims
            normalized_key = claim_text.lower()

            if normalized_key in seen_claims:
                continue

            seen_claims.add(normalized_key)

            claims.append({
                "claim_id": len(claims) + 1,
                "claim_text": claim_text,
                "context": cleaned_answer,
                "type": "factual_candidate"
            })

    return claims


if __name__ == "__main__":

    sample_answer = """
    1. Jupiter is the largest planet in the Solar System.
    2. Earth is the third planet from the Sun and the largest
       of the terrestrial planets.
    3. Saturn is famous for its prominent ring system.
    """

    extracted = extract_claims(sample_answer)

    print("\n========== CLAIM EXTRACTION TEST ==========\n")

    for claim in extracted:
        print(f"Claim ID: {claim['claim_id']}")
        print(f"Claim: {claim['claim_text']}")
        print()
