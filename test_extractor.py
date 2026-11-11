from core.claim_extractor import extract_claims


answer = """
Artificial Intelligence is a branch of computer science.

Machine learning is a subset of artificial intelligence.

The Earth is the largest planet in our solar system.

Python was first released in 1991.

I hope this explanation helps!
"""

claims = extract_claims(answer)

print("\nEXTRACTED CLAIMS\n")

for claim in claims:
    print(f"Claim ID: {claim['claim_id']}")
    print(f"Claim: {claim['claim_text']}")
    print(f"Type: {claim['type']}")
    print("-" * 50)