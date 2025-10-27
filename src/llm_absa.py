import ollama
import json
from typing import List
from src.base import ABSAAnalyzer, AspectSentiment


class LLMABSA(ABSAAnalyzer):
    def __init__(self, model="llama2"):
        self.model = model

    def analyze(self, text: str) -> List[AspectSentiment]:
        prompt = self._create_prompt(text)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json'  # Request JSON output
            )

            result = json.loads(response['message']['content'])
            return self._parse_response(result)

        except Exception as e:
            print(f"Error: {e}")
            return []


    def _create_prompt(self, text: str) -> str:
        prompt = f"""You are an aspect-based sentiment analyzer. Analyze the following text and extract:
1. All aspects (features, entities, topics) mentioned
2. The sentiment toward each aspect (positive, negative, or neutral)
3. A confidence score (0.0 to 1.0)

Text: "{text}"

Return your analysis as a JSON object with this structure:
{{
    "aspects": [
        {{
            "aspect": "aspect name",
            "sentiment": "positive|negative|neutral",
            "confidence": 0.95
        }}
    ]
}}

Be specific and only extract aspects that are explicitly mentioned. Provide your response ONLY as valid JSON, no additional text."""

        return prompt

    def _parse_response(self, result: dict) -> List[AspectSentiment]:
        aspects_list = []

        for item in result.get('aspects', []):
            aspects_list.append(AspectSentiment(
                aspect=item['aspect'],
                sentiment=item['sentiment'].lower(),
                confidence=float(item['confidence']),
                text_span=None
            ))

        return aspects_list