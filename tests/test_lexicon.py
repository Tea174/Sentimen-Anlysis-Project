# testing Implementation 1 - Lexicon-Based with spaCy
from src.lexicon_absa import LexiconABSA
def test_lexicon_absa():
    analyzer = LexiconABSA()
    text = "The pizza was delicious but the service was terrible."
    results = analyzer.analyze(text)

    print(f"\nAnalyzing: '{text}'")
    for result in results:
        print(result)

    assert len(results) > 0


if __name__ == "__main__":
    test_lexicon_absa()
