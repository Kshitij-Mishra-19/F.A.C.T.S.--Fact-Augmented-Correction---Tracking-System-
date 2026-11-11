
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from knowledge_base.retrieval import retrieve_evidence


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Loading NLI model on: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()
model.to(device)

ID_TO_LABEL = {
    int(index): label.lower()
    for index, label in model.config.id2label.items()
}

print("MODEL LABEL MAPPING:")
print(ID_TO_LABEL)


# ============================================================
# 2. EXPERIMENTAL THRESHOLDS
# ============================================================

RELEVANCE_THRESHOLD = 0.30

ENTAILMENT_THRESHOLD = 0.80

CONTRADICTION_THRESHOLD = 0.90


# Generic words that should not dominate relevance matching.
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on",
    "at", "for", "from", "by", "with", "and", "or",
    "but", "that", "this", "these", "those",
    "it", "its", "as", "has", "have", "had",
    "do", "does", "did", "can", "could", "would",
    "should", "will", "may", "might",
    "solar", "system", "planet"
}


# ============================================================
# 3. TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if not isinstance(text, str):
        return ""

    text = text.strip()

    # Remove repeated numbering such as:
    # 2. 2. Earth is a planet.
    previous = None

    while previous != text:

        previous = text

        text = re.sub(
            r"^\s*(?:\d+[\.\)]\s*|[-*]\s*)",
            "",
            text
        )

    # Normalize case and punctuation.
    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. KEYWORD OVERLAP / EVIDENCE RELEVANCE
# ============================================================

def get_content_words(text):

    normalized = normalize_text(text)

    words = set(normalized.split())

    return {
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    }


def calculate_keyword_overlap(evidence_text, claim_text):

    evidence_words = get_content_words(evidence_text)

    claim_words = get_content_words(claim_text)

    if not evidence_words or not claim_words:
        return 0.0

    intersection = evidence_words.intersection(claim_words)

    union = evidence_words.union(claim_words)

    return len(intersection) / len(union)


def is_exact_match(evidence_text, claim_text):

    return (
        normalize_text(evidence_text)
        == normalize_text(claim_text)
    )


# ============================================================
# 5. NLI CLASSIFICATION
# ============================================================

def classify_pair(premise, hypothesis):

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    scores = {
        "entailment": 0.0,
        "contradiction": 0.0,
        "neutral": 0.0
    }

    for index, probability in enumerate(probabilities):

        label = ID_TO_LABEL[index]

        for category in scores:

            if category in label:

                scores[category] = float(probability)

                break

    return scores


# ============================================================
# 6. VERIFY A SINGLE CLAIM
# ============================================================

def verify_single_claim(claim_text, top_k=5):

    if not isinstance(claim_text, str) or not claim_text.strip():

        return {
            "claim": claim_text,
            "status": "insufficient_evidence",
            "reason": "Invalid or empty claim.",
            "evidence": [],
            "scores": {}
        }

    # --------------------------------------------------------
    # Retrieve evidence
    # --------------------------------------------------------

    evidence_list = retrieve_evidence(
        claim_text,
        top_k=top_k
    )

    if not evidence_list:

        return {
            "claim": claim_text,
            "status": "insufficient_evidence",
            "reason": "No evidence was retrieved.",
            "evidence": [],
            "scores": {}
        }

    comparisons = []

    # --------------------------------------------------------
    # Compare evidence with claim
    # --------------------------------------------------------

    for evidence in evidence_list:

        evidence_text = evidence["text"]

        exact_match = is_exact_match(
            evidence_text,
            claim_text
        )

        keyword_overlap = calculate_keyword_overlap(
            evidence_text,
            claim_text
        )

        relevant = (
            exact_match
            or keyword_overlap >= RELEVANCE_THRESHOLD
        )

        # Run NLI only on evidence that passes relevance filtering.
        if relevant:

            scores = classify_pair(
                premise=evidence_text,
                hypothesis=claim_text
            )

        else:

            scores = {
                "entailment": 0.0,
                "contradiction": 0.0,
                "neutral": 1.0
            }

        comparisons.append({

            "evidence": evidence,

            "scores": scores,

            "keyword_overlap": keyword_overlap,

            "exact_match": exact_match,

            "relevant": relevant

        })

    # --------------------------------------------------------
    # DEBUGGING OUTPUT
    # --------------------------------------------------------

    print("\n========== EVIDENCE DEBUG ==========")

    for item in comparisons:

        print("\nEvidence:", item["evidence"]["text"])

        print(
            "Similarity:",
            item["evidence"]["similarity"]
        )

        print(
            "Keyword overlap:",
            round(item["keyword_overlap"], 4)
        )

        print(
            "Relevant:",
            item["relevant"]
        )

        print(
            "Exact match:",
            item["exact_match"]
        )

        print(
            "NLI Scores:",
            item["scores"]
        )

    print("\n====================================")

    # --------------------------------------------------------
    # Keep only relevant evidence for decisions
    # --------------------------------------------------------

    relevant_comparisons = [
        item
        for item in comparisons
        if item["relevant"]
    ]

    if not relevant_comparisons:

        status = "insufficient_evidence"

        reason = (
            "Retrieved passages did not contain enough "
            "relevant information to evaluate the claim."
        )

        entailment_score = 0.0

        contradiction_score = 0.0

    else:

        # ----------------------------------------------------
        # Find strongest supporting and contradicting evidence
        # ----------------------------------------------------

        best_entailment = max(
            relevant_comparisons,
            key=lambda item: item["scores"]["entailment"]
        )

        best_contradiction = max(
            relevant_comparisons,
            key=lambda item: item["scores"]["contradiction"]
        )

        entailment_score = best_entailment["scores"]["entailment"]

        contradiction_score = best_contradiction["scores"]["contradiction"]

        exact_support = any(
            item["exact_match"]
            for item in relevant_comparisons
        )

        strong_support = (
            exact_support
            or entailment_score >= ENTAILMENT_THRESHOLD
        )

        strong_contradiction = (
            contradiction_score >= CONTRADICTION_THRESHOLD
        )

        # ----------------------------------------------------
        # Conflict-aware decision
        # ----------------------------------------------------

        if strong_support and strong_contradiction:

            status = "insufficient_evidence"

            reason = (
                "Retrieved evidence contains both strong "
                "support and strong contradiction. "
                "The conflict requires further investigation."
            )

        elif strong_support:

            status = "supported"

            if exact_support:

                reason = (
                    "An exact matching statement was found "
                    "in the knowledge base."
                )

            else:

                reason = (
                    "Relevant evidence strongly supports "
                    "the claim according to the NLI model."
                )

        elif strong_contradiction:

            status = "contradicted"

            reason = (
                "Relevant evidence strongly contradicts "
                "the claim according to the NLI model."
            )

        else:

            status = "insufficient_evidence"

            reason = (
                "Relevant evidence was found, but it does "
                "not strongly support or contradict the claim."
            )

    # --------------------------------------------------------
    # Format evidence for output
    # --------------------------------------------------------

    formatted_evidence = []

    for item in comparisons:

        evidence = item["evidence"]

        scores = item["scores"]

        formatted_evidence.append({

            "text": evidence["text"],

            "source": evidence["source"],

            "similarity": evidence["similarity"],

            "keyword_overlap": round(
                item["keyword_overlap"], 4
            ),

            "exact_match": item["exact_match"],

            "relevant": item["relevant"],

            "nli_scores": {
                key: round(value, 4)
                for key, value in scores.items()
            }

        })

    return {

        "claim": claim_text,

        "status": status,

        "reason": reason,

        "scores": {

            "entailment": round(entailment_score, 4),

            "contradiction": round(contradiction_score, 4)

        },

        "evidence": formatted_evidence

    }


# ============================================================
# 7. VERIFY MULTIPLE CLAIMS
# ============================================================

def verify_claims(claims):

    results = []

    for claim in claims:

        result = verify_single_claim(
            claim["claim_text"]
        )

        result["claim_id"] = claim["claim_id"]

        results.append(result)

    total = len(results)

    supported = sum(
        item["status"] == "supported"
        for item in results
    )

    contradicted = sum(
        item["status"] == "contradicted"
        for item in results
    )

    insufficient = sum(
        item["status"] == "insufficient_evidence"
        for item in results
    )

    coverage = (
        supported / total
        if total > 0
        else 0.0
    )

    return {

        "total_claims": total,

        "supported_count": supported,

        "contradicted_count": contradicted,

        "insufficient_count": insufficient,

        "evidence_coverage": round(coverage, 4),

        "claims": results

    }


# ============================================================
# 8. DIRECT TESTING
# ============================================================

if __name__ == "__main__":

    print("\n========== NLI MODEL DIAGNOSTICS ==========")

    test_cases = [

        (
            "Jupiter is the largest planet in the Solar System.",
            "Jupiter is the largest planet in the Solar System."
        ),

        (
            "Saturn is famous for its prominent ring system.",
            "Saturn has a prominent ring system."
        ),

        (
            "Jupiter is the largest planet in the Solar System.",
            "Earth is the largest planet in the Solar System."
        )

    ]

    for premise, hypothesis in test_cases:

        print("\nPremise:", premise)

        print("Hypothesis:", hypothesis)

        print(
            "Scores:",
            classify_pair(premise, hypothesis)
        )
