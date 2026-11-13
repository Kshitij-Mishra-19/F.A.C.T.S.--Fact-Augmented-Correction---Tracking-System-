
from core.pipeline import run_pipeline

test_answer = """
Mercury is the largest planet in the Solar System.
"""

result = run_pipeline(test_answer)

print("\n========== CONTRADICTION TEST ==========")

print("\nVerification:")
print(result["verification"])

print("\nReview:")
print(result["review"])

print("========================================")
