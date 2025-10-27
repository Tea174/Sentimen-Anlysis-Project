from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
import sys
import re

sys.path.insert(0, '.')
from src.base import ABSAAnalyzer, AspectSentiment
from src.utils import AspectExtractionMixin


class TransformerABSA(AspectExtractionMixin, ABSAAnalyzer):
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        AspectExtractionMixin.__init__(self)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def analyze(self, text: str) -> List[AspectSentiment]:
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        doc = self.nlp(text)

        # Extract raw candidates
        candidates = self._extract_aspects(doc)  # From mixin

        # Merge coordinated aspects like "cookies and creme ice cream"
        merged = self._merge_coordinated_aspects(candidates, doc, text)

        # Normalize & dedupe
        normalized = []
        seen = set()
        for c in merged:
            norm = self._normalize_aspect(self._get_text(c)).lower()
            if norm not in seen and norm:
                normalized.append(c)
                seen.add(norm)

        results = []
        for aspect in normalized:
            normalized_text = self._normalize_aspect(self._get_text(aspect))
            sentiment_info = self._classify_aspect_sentiment(text, normalized_text)
            if sentiment_info:
                results.append(AspectSentiment(
                    aspect=normalized_text,
                    sentiment=sentiment_info['label'],
                    confidence=sentiment_info['score'],
                    text_span=(self._get_start_char(aspect), self._get_end_char(aspect))
                ))

        return results

    def _merge_coordinated_aspects(self, aspects, doc, raw_text):
        """Merge adjacent aspects connected by 'and'/',' - simplified for Transformer"""
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

    def _classify_aspect_sentiment(self, text: str, aspect: str):
        inputs = self.tokenizer(
            text,
            aspect,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        return {
            'label': self.id2label[prediction],
            'score': confidence
        }