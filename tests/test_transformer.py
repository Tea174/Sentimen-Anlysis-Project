
# Test Implementation 2 - Pre-trained Model
from src.transformer_absa import TransformerABSA
def test_transformer_absa():
    analyzer = TransformerABSA()
    text = "The pizza was delicious but the service was terrible."
    results = analyzer.analyze(text)

    print(f"\nAnalyzing: '{text}'")
    for result in results:
        print(result)


if __name__ == "__main__":
    test_transformer_absa()