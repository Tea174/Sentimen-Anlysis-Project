


# Test Implementation 3 - LLM with Ollama
from src.llm_absa import LLMABSA
def test_llm_absa():
    analyzer = LLMABSA()
    text = "The pizza was delicious but the service was terrible."
    results = analyzer.analyze(text)

    print(f"\nAnalyzing: '{text}'")
    for result in results:
        print(result)


if __name__ == "__main__":
    test_llm_absa()