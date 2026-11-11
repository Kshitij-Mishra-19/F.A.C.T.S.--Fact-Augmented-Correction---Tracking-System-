from core.claim_extractor import extract_claims
from knowledge_base.retrieval import retrieve_evidence


answer = """
Jupiter is the largest planet in our solar system.

Earth is the third planet from the Sun.

Earth is the largest planet in our solar system.
"""

claims = extract_claims(answer)

for claim in claims:

    print("\nCLAIM:", claim["claim_text"])

    evidence = retrieve_evidence(
        claim["claim_text"],
        top_k=2
    )

    if not evidence:
        print("No evidence found.")
        continue

    for item in evidence:

        print("Evidence:", item["text"])
        print("Source:", item["source"])
        print("Similarity:", item["similarity"])