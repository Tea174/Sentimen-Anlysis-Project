from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
from src.base import ABSAAnalyzer, AspectSentiment
import spacy


class TransformerABSA(ABSAAnalyzer):
    def __init__(self, model_name="yangheng/deberta-v3-base-absa-v1.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Use spaCy for aspect extraction
        self.nlp = spacy.load("en_core_web_sm")

        # Sentiment labels (check model config for actual labels)
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def analyze(self, text: str) -> List[AspectSentiment]:
        # Extract candidate aspects
        doc = self.nlp(text)
        aspects = self._extract_aspects(doc)

        results = []
        for aspect in aspects:
            sentiment_info = self._classify_aspect_sentiment(text, aspect.text)
            if sentiment_info:
                results.append(AspectSentiment(
                    aspect=aspect.text,
                    sentiment=sentiment_info['label'],
                    confidence=sentiment_info['score'],
                    text_span=(aspect.start_char, aspect.end_char)
                ))

        return results

    def _extract_aspects(self, doc):
        """Extract aspect candidates (nouns and noun phrases)"""
        aspects = []
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                aspects.append(chunk)
        return aspects

    def _classify_aspect_sentiment(self, text: str, aspect: str):
        """Classify sentiment for a specific aspect"""
        # Format: "[CLS] text [SEP] aspect [SEP]"
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