
from integrations.generators.base_generator import BaseGenerator


class MockGenerator(BaseGenerator):

    def __init__(self):

        self.call_count = 0

    def generate(self, prompt: str) -> str:

        self.call_count += 1

        # First response intentionally contains a false claim
        if self.call_count == 1:

            return """
            Jupiter is the largest planet in the Solar System.

            Earth is the largest planet in the Solar System.

            Saturn has a prominent ring system.
            """

        # Second response is corrected
        return """
        Jupiter is the largest planet in the Solar System.

        Earth is the third planet from the Sun.

        Saturn has a prominent ring system.
        """
