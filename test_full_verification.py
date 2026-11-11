from core.claim_extractor import extract_claims
from core.claim_verifier import verify_claims


answer = """
Jupiter is the largest planet in the Solar System.

Earth is the third planet from the Sun.

Earth is the largest planet in the Solar System.

Saturn has a prominent ring system.
"""

# Step 1: Extract claims
claims = extract_claims(answer)

# Step 2: Verify each claim
result = verify_claims(claims)

# Step 3: Display results
print("\n========== HALLUGUARD VERIFICATION ==========\n")

for claim in result["claims"]:

    print(f"Claim ID: {claim['claim_id']}")
    print(f"Claim: {claim['claim']}")
    print(f"Status: {claim['status']}")
    print(f"Reason: {claim['reason']}")

    print("\nEvidence:")

    for evidence in claim["evidence"]:
        print(f"- {evidence['text']}")
        print(f"  Source: {evidence['source']}")

    print("-" * 60)


print("\n========== SUMMARY ==========")

print("Total claims:", result["total_claims"])
print("Supported:", result["supported_count"])
print("Contradicted:", result["contradicted_count"])
print("Insufficient:", result["insufficient_count"])
print("Evidence coverage:", result["evidence_coverage"])