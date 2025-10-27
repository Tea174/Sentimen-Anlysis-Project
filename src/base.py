from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from abc import ABC, abstractmethod
import time
import sys
import numpy as np


@dataclass
class AspectSentiment:
    """Data class for aspect-sentiment pairs"""
    aspect: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    text_span: Optional[Tuple[int, int]] = None

    def __str__(self):
        return f"Aspect: '{self.aspect}' → Sentiment: {self.sentiment.upper()} (confidence: {self.confidence:.2f})"


class ABSAAnalyzer(ABC):
    @abstractmethod
    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs

        Args:
            text: Input text to analyze

        Returns:
            List of AspectSentiment objects
        """
        pass

    # ==================== PERFORMANCE METRICS ====================

    def calculate_speed(self, texts: List[str]) -> Dict[str, float]:
        """Calculate processing speed metrics for the model."""
        if not texts:
            return {
                'total_time': 0.0,
                'avg_time_per_text': 0.0,
                'throughput_texts_per_second': 0.0,
                'texts_processed': 0
            }

        start_time = time.time()
        for text in texts:
            self.analyze(text)
        total_time = time.time() - start_time

        return {
            'total_time': total_time,
            'avg_time_per_text': total_time / len(texts),
            'throughput_texts_per_second': len(texts) / total_time if total_time > 0 else 0.0,
            'texts_processed': len(texts)
        }

    def calculate_aspects_detected(self, texts: List[str]) -> Dict[str, float]:
        """Calculate statistics on number of aspects detected."""
        aspect_counts = [len(self.analyze(text)) for text in texts]

        if not aspect_counts:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'total': 0
            }

        return {
            'mean': float(np.mean(aspect_counts)),
            'median': float(np.median(aspect_counts)),
            'std': float(np.std(aspect_counts)),
            'min': int(min(aspect_counts)),
            'max': int(max(aspect_counts)),
            'total': int(sum(aspect_counts))
        }

    def calculate_avg_confidence(self, texts: List[str]) -> Dict[str, float]:
        """Calculate confidence score statistics."""
        confidences = []
        for text in texts:
            results = self.analyze(text)
            confidences.extend([r.confidence for r in results])

        if not confidences:
            return {
                'avg': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0,
                'total_aspects': 0
            }

        return {
            'avg': float(np.mean(confidences)),
            'min': float(min(confidences)),
            'max': float(max(confidences)),
            'std': float(np.std(confidences)),
            'total_aspects': len(confidences)
        }

    def calculate_memory_usage(self, texts: List[str]) -> Dict[str, float]:
        """
        Calculate memory usage metrics.
        Override in subclass for model-specific memory tracking.
        """
        import gc
        gc.collect()

        memory_samples = []
        for text in texts:
            results = self.analyze(text)
            results_size = sys.getsizeof(results) / (1024 * 1024)  # Convert to MB
            memory_samples.append(results_size)

        return {
            'peak_memory_mb': float(max(memory_samples)) if memory_samples else 0.0,
            'avg_memory_per_text_mb': float(np.mean(memory_samples)) if memory_samples else 0.0,
            'total_memory_mb': float(sum(memory_samples)) if memory_samples else 0.0
        }

    def calculate_initialization_time(self) -> float:
        """
        Calculate model initialization time.
        Override in subclass for more accurate model-specific timing.
        """
        start = time.time()
        _ = self.__class__()
        return time.time() - start

    def calculate_all_metrics(self, texts: List[str]) -> Dict[str, Dict]:
        """
        Calculate all performance metrics at once.

        Args:
            texts: List of text strings to analyze

        Returns:
            Dictionary containing all metrics organized by category
        """
        return {
            'speed': self.calculate_speed(texts),
            'aspects_detected': self.calculate_aspects_detected(texts),
            'avg_confidence': self.calculate_avg_confidence(texts),
            'memory_usage': self.calculate_memory_usage(texts),
            'initialization': {'time_seconds': self.calculate_initialization_time()}
        }

    def print_metrics_report(self, metrics: Dict[str, Dict]) -> None:
        """
        Print a formatted report of all metrics.

        Args:
            metrics: Dictionary from calculate_all_metrics()
        """
        print("=" * 60)
        print(f"PERFORMANCE REPORT: {self.__class__.__name__}")
        print("=" * 60)

        print("\n📊 SPEED METRICS")
        print(f"  Total time: {metrics['speed']['total_time']:.4f}s")
        print(f"  Avg time per text: {metrics['speed']['avg_time_per_text']:.4f}s")
        print(f"  Throughput: {metrics['speed']['throughput_texts_per_second']:.2f} texts/sec")

        print("\n🎯 ASPECTS DETECTED")
        print(f"  Mean: {metrics['aspects_detected']['mean']:.2f}")
        print(f"  Median: {metrics['aspects_detected']['median']:.2f}")
        print(f"  Std Dev: {metrics['aspects_detected']['std']:.2f}")
        print(f"  Range: [{metrics['aspects_detected']['min']}, {metrics['aspects_detected']['max']}]")
        print(f"  Total: {metrics['aspects_detected']['total']}")

        print("\n💯 CONFIDENCE METRICS")
        print(f"  Avg confidence: {metrics['avg_confidence']['avg']:.4f}")
        print(f"  Range: [{metrics['avg_confidence']['min']:.4f}, {metrics['avg_confidence']['max']:.4f}]")
        print(f"  Std Dev: {metrics['avg_confidence']['std']:.4f}")

        print("\n💾 MEMORY USAGE")
        print(f"  Peak memory: {metrics['memory_usage']['peak_memory_mb']:.4f} MB")
        print(f"  Avg per text: {metrics['memory_usage']['avg_memory_per_text_mb']:.4f} MB")

        print("\n⚡ INITIALIZATION")
        print(f"  Time: {metrics['initialization']['time_seconds']:.4f}s")

        print("\n" + "=" * 60)