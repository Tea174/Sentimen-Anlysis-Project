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