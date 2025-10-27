import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Set
from src.base import ABSAAnalyzer, AspectSentiment


class LexiconABSA(ABSAAnalyzer):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        aspects = self._extract_aspects(doc)
        results = []

        for aspect in aspects:
            sentiment_info = self._get_aspect_sentiment(aspect, doc)
            if sentiment_info:
                results.append(sentiment_info)

        return results

    def _extract_aspects(self, doc) -> List:
        """Extract aspect terms (nouns and noun phrases)"""
        aspects = []

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            # Filter out pronouns and determiners
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                aspects.append(chunk)

        # Also get standalone nouns not in chunks
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not any(token in chunk for chunk in aspects):
                aspects.append(token)

        return aspects

    def _get_aspect_sentiment(self, aspect, doc):
        """Find opinion words related to aspect using dependency parsing"""
        # Get the root token
        aspect_token = aspect.root if hasattr(aspect, 'root') else aspect

        # Find related opinion words through dependencies
        opinion_words = []

        # Check modifiers (amod, advmod)
        for child in aspect_token.children:
            if child.dep_ in ['amod', 'advmod', 'acomp']:
                opinion_words.append(child)

        # Check if aspect is subject of a verb
        if aspect_token.dep_ in ['nsubj', 'nsubjpass']:
            verb = aspect_token.head
            # Get complements and modifiers of the verb
            for child in verb.children:
                if child.dep_ in ['acomp', 'xcomp', 'advmod']:
                    opinion_words.append(child)

        # Check if aspect is object of a verb
        if aspect_token.dep_ in ['dobj', 'pobj']:
            verb = aspect_token.head
            opinion_words.append(verb)

        # Build context string for sentiment analysis
        if opinion_words:
            context = ' '.join([w.text for w in opinion_words])
        else:
            # Use surrounding window if no direct dependencies
            start = max(0, aspect_token.i - 3)
            end = min(len(doc), aspect_token.i + 4)
            context = ' '.join([doc[i].text for i in range(start, end)])

        # Handle negations
        is_negated = self._check_negation(aspect_token)

        # Get sentiment score
        scores = self.vader.polarity_scores(context)
        compound = scores['compound']

        # Flip if negated
        if is_negated:
            compound = -compound

        # Classify sentiment
        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Calculate confidence
        confidence = abs(compound)

        return AspectSentiment(
            aspect=aspect.text,
            sentiment=sentiment,
            confidence=confidence,
            text_span=(aspect.start_char, aspect.end_char)
        )

    def _check_negation(self, token) -> bool:
        """Check if token is negated"""
        # Check for negation in children
        for child in token.children:
            if child.dep_ == 'neg':
                return True

        # Check for negation in ancestors
        for ancestor in token.ancestors:
            for child in ancestor.children:
                if child.dep_ == 'neg' and child.i < token.i:
                    return True

        return False