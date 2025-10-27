import re
from typing import List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys

sys.path.insert(0, '.')
from src.base import ABSAAnalyzer, AspectSentiment
from src.utils import AspectExtractionMixin


class LexiconABSA(AspectExtractionMixin, ABSAAnalyzer):
    def __init__(self):
        AspectExtractionMixin.__init__(self)
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> List[AspectSentiment]:
        # small cleanup
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        doc = self.nlp(text)

        # extract raw candidates
        candidates = self._extract_aspects(doc)  # From mixin

        # try merging candidates that appear together with "and" / "," FIRST
        merged = self._merge_coordinated_aspects(candidates, doc, text)

        # THEN normalize & dedupe
        normalized = []
        seen = set()
        for c in merged:
            norm = self._normalize_aspect(self._get_text(c)).lower()  # From mixin
            if norm not in seen and norm:
                normalized.append(c)
                seen.add(norm)

        print(
            f"DEBUG: Found {len(normalized)} aspects: {[self._normalize_aspect(self._get_text(a)) for a in normalized]}")

        results = []
        for aspect in normalized:
            sentiment_info = self._get_aspect_sentiment(aspect, doc)
            if sentiment_info:
                results.append(sentiment_info)

        return results

    # -----------------------
    # Merge coordinated aspects (Lexicon-specific)
    # -----------------------
    def _merge_coordinated_aspects(self, aspects, doc, raw_text):
        """Merge adjacent aspects connected by 'and'/',' into single phrases."""
        if not aspects:
            return []

        items = []
        for a in aspects:
            items.append((self._get_start_char(a), self._get_end_char(a), a))

        items.sort(key=lambda x: x[0])

        merged = []
        skip_next = False

        for i in range(len(items)):
            if skip_next:
                skip_next = False
                continue

            start_i, end_i, a_i = items[i]
            merged_flag = False

            if i + 1 < len(items):
                start_j, end_j, a_j = items[i + 1]
                mid_sub = raw_text[end_i:start_j].lower()

                if re.search(r'\b(and|,|&)\b', mid_sub) and len(mid_sub) < 10:
                    candidate = raw_text[start_i:end_j].strip()
                    candidate_norm = self._normalize_aspect(candidate)

                    single_i = self._normalize_aspect(self._get_text(a_i)).lower()
                    single_j = self._normalize_aspect(self._get_text(a_j)).lower()

                    if (single_i in candidate_norm.lower() and
                            single_j in candidate_norm.lower() and
                            len(candidate_norm) > max(len(single_i), len(single_j))):

                        try:
                            token_start = None
                            token_end = None
                            for tok in doc:
                                if tok.idx >= start_i and token_start is None:
                                    token_start = tok.i
                                if tok.idx + len(tok.text) <= end_j:
                                    token_end = tok.i + 1

                            if token_start is not None and token_end is not None and token_end > token_start:
                                span = doc[token_start:token_end]
                                merged.append(span)
                                merged_flag = True
                                skip_next = True
                        except Exception:
                            merged.append(self.nlp(candidate_norm))
                            merged_flag = True
                            skip_next = True

            if not merged_flag:
                merged.append(a_i)

        final = []
        seen_exact = set()

        for a in merged:
            norm = self._normalize_aspect(self._get_text(a)).lower()
            if norm in seen_exact:
                continue
            seen_exact.add(norm)
            final.append(a)

        return final

    # -----------------------
    # Sentiment extraction (Lexicon-specific)
    # -----------------------
    def _get_aspect_sentiment(self, aspect, doc):
        aspect_root = self._get_root(aspect)
        opinion_tokens = []

        for child in aspect_root.children:
            if child.dep_ in {'amod', 'advmod', 'acomp', 'acls'}:
                opinion_tokens.append(child)

        if aspect_root.dep_ in {'nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr'}:
            head = aspect_root.head
            for child in head.children:
                if child.dep_ in {'acomp', 'xcomp', 'advmod', 'attr', 'dobj'}:
                    opinion_tokens.append(child)

        if opinion_tokens:
            opinion_tokens = sorted(set(opinion_tokens), key=lambda t: t.i)
            context = ' '.join([t.text for t in opinion_tokens])
        else:
            context = aspect_root.sent.text.strip()

        is_negated = self._check_negation(aspect_root)

        scores = self.vader.polarity_scores(context)
        compound = scores['compound']
        if is_negated:
            compound = -compound

        if compound >= 0.05:
            sentiment = 'positive'
        elif compound <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        confidence = abs(compound)
        normalized_aspect = self._normalize_aspect(self._get_text(aspect))

        return AspectSentiment(
            aspect=normalized_aspect,
            sentiment=sentiment,
            confidence=confidence,
            text_span=(self._get_start_char(aspect), self._get_end_char(aspect))
        )

    def _check_negation(self, token) -> bool:
        for child in token.children:
            if child.dep_ == 'neg' or child.lower_ in {'no', 'not', "n't", 'never', 'none'}:
                return True

        for anc in token.ancestors:
            for c in anc.children:
                if c.dep_ == 'neg' or c.lower_ in {'no', 'not', "n't", 'never', 'none'}:
                    return True

        doc = token.doc
        start = max(0, token.i - 4)
        for i in range(start, token.i):
            if doc[i].lower_ in {'no', 'not', "n't", 'never', 'none'}:
                return True

        return False