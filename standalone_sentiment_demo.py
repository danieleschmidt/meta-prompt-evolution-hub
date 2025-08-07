"""
Standalone Sentiment Analyzer Demo
Pure Python implementation with no external dependencies
"""

import random
import time
import json
from typing import Dict, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass

class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"

@dataclass
class SentimentResult:
    text: str
    label: SentimentLabel
    confidence: float
    processing_time: float
    method: str = "rule-based"

class StandaloneSentimentAnalyzer:
    """Pure Python sentiment analyzer with no dependencies"""
    
    def __init__(self):
        # Expanded sentiment lexicon
        self.positive_words = {
            'love', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect', 
            'outstanding', 'brilliant', 'awesome', 'superb', 'magnificent', 'terrific',
            'good', 'nice', 'happy', 'pleased', 'satisfied', 'delighted', 'thrilled',
            'best', 'favorite', 'recommend', 'impressed', 'beautiful', 'incredible',
            'marvelous', 'exceptional', 'phenomenal', 'remarkable', 'splendid'
        }
        
        self.negative_words = {
            'hate', 'terrible', 'awful', 'horrible', 'disgusting', 'worst', 'bad',
            'disappointing', 'poor', 'sad', 'angry', 'frustrated', 'annoyed',
            'useless', 'broken', 'failed', 'wrong', 'problem', 'issue', 'concern',
            'slow', 'expensive', 'cheap', 'ugly', 'difficult', 'hard', 'impossible',
            'disaster', 'nightmare', 'pathetic', 'ridiculous', 'unacceptable'
        }
        
        # Sentiment modifiers
        self.amplifiers = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'completely'}
        self.negators = {'not', 'no', 'never', 'nothing', 'none', 'neither', "don't", "doesn't", "won't", "can't"}
        
        # Analysis stats
        self.total_analyses = 0
        self.processing_times = []
    
    def analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyze sentiment using rule-based approach"""
        start_time = time.time()
        self.total_analyses += 1
        
        if not isinstance(text, str) or not text.strip():
            return SentimentResult(
                text=text,
                label=SentimentLabel.NEUTRAL,
                confidence=0.0,
                processing_time=time.time() - start_time,
                method="error-fallback"
            )
        
        # Preprocessing
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_score = 0
        negative_score = 0
        
        # Analyze each word with context
        for i, word in enumerate(words):
            # Check for negation in previous words
            negated = False
            for j in range(max(0, i-3), i):
                if words[j] in self.negators:
                    negated = True
                    break
            
            # Check for amplifiers in previous words
            amplified = False
            for j in range(max(0, i-2), i):
                if words[j] in self.amplifiers:
                    amplified = True
                    break
            
            # Base score
            base_score = 1.0
            if amplified:
                base_score = 1.5
            
            # Apply sentiment scoring
            if word in self.positive_words:
                if negated:
                    negative_score += base_score
                else:
                    positive_score += base_score
            elif word in self.negative_words:
                if negated:
                    positive_score += base_score
                else:
                    negative_score += base_score
        
        # Determine sentiment
        if positive_score > negative_score:
            label = SentimentLabel.POSITIVE
            confidence = min(0.95, 0.6 + (positive_score - negative_score) * 0.1)
        elif negative_score > positive_score:
            label = SentimentLabel.NEGATIVE
            confidence = min(0.95, 0.6 + (negative_score - positive_score) * 0.1)
        else:
            label = SentimentLabel.NEUTRAL
            confidence = 0.5 + random.uniform(-0.1, 0.1)  # Add some variance
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        return SentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze multiple texts"""
        return [self.analyze_sentiment(text) for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        if not self.processing_times:
            return {"message": "No analyses performed yet"}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        min_time = min(self.processing_times)
        max_time = max(self.processing_times)
        
        return {
            "total_analyses": self.total_analyses,
            "avg_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "throughput_per_second": 1.0 / avg_time if avg_time > 0 else 0,
            "vocabulary_size": len(self.positive_words) + len(self.negative_words)
        }

def demo_basic_functionality():
    """Demonstrate basic sentiment analysis"""
    print("🎯 Basic Sentiment Analysis Demo")
    print("-" * 40)
    
    analyzer = StandaloneSentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this amazing product!",
        "This is the worst service I've ever experienced",
        "It's an okay product, nothing special",
        "Fantastic quality and excellent customer service!",
        "Poor design and very disappointing results",
        "The item is fine, meets basic expectations",
        "Not bad, but could be much better",
        "Incredible value for money, highly recommend!",
        "I don't like this at all, very frustrating",
        "Average performance, works as expected"
    ]
    
    print(f"Analyzing {len(test_texts)} sample texts:\n")
    
    results = []
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_sentiment(text)
        results.append(result)
        
        # Format output
        emoji = "😊" if result.label == SentimentLabel.POSITIVE else "😢" if result.label == SentimentLabel.NEGATIVE else "😐"
        print(f"{i:2d}. {emoji} {result.label.value.upper()} ({result.confidence:.2f})")
        print(f"    Text: \"{text}\"")
        print(f"    Time: {result.processing_time*1000:.1f}ms\n")
    
    return results, analyzer.get_stats()

def demo_batch_processing():
    """Demonstrate batch processing"""
    print("⚡ Batch Processing Demo")
    print("-" * 40)
    
    analyzer = StandaloneSentimentAnalyzer()
    
    # Generate test data
    sample_reviews = [
        "Great product, fast shipping!",
        "Terrible quality, waste of money",
        "It's okay, average quality",
        "Outstanding service and support",
        "Very disappointed with purchase",
        "Good value for the price",
        "Excellent build quality",
        "Poor customer service experience",
        "Meets my expectations perfectly",
        "Would not recommend to others"
    ] * 10  # 100 total reviews
    
    print(f"Processing {len(sample_reviews)} reviews in batch...\n")
    
    start_time = time.time()
    results = analyzer.batch_analyze(sample_reviews)
    total_time = time.time() - start_time
    
    # Analyze results
    sentiment_counts = {
        SentimentLabel.POSITIVE: 0,
        SentimentLabel.NEGATIVE: 0,
        SentimentLabel.NEUTRAL: 0
    }
    
    confidence_sum = 0
    for result in results:
        sentiment_counts[result.label] += 1
        confidence_sum += result.confidence
    
    avg_confidence = confidence_sum / len(results)
    throughput = len(results) / total_time
    
    print(f"📊 Batch Results:")
    print(f"  Positive: {sentiment_counts[SentimentLabel.POSITIVE]:3d} ({sentiment_counts[SentimentLabel.POSITIVE]/len(results)*100:.1f}%)")
    print(f"  Negative: {sentiment_counts[SentimentLabel.NEGATIVE]:3d} ({sentiment_counts[SentimentLabel.NEGATIVE]/len(results)*100:.1f}%)")
    print(f"  Neutral:  {sentiment_counts[SentimentLabel.NEUTRAL]:3d} ({sentiment_counts[SentimentLabel.NEUTRAL]/len(results)*100:.1f}%)")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    print(f"  Processing Time: {total_time:.3f}s")
    print(f"  Throughput: {throughput:.1f} texts/second")
    
    return results

def demo_edge_cases():
    """Test edge cases and robustness"""
    print("🛡️ Edge Cases & Robustness Demo")
    print("-" * 40)
    
    analyzer = StandaloneSentimentAnalyzer()
    
    edge_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("a", "Single character"),
        ("!!!", "Special characters only"),
        ("12345", "Numbers only"),
        ("This is not bad", "Negation handling"),
        ("I don't hate this", "Double negation"),
        ("Very very very good", "Multiple amplifiers"),
        ("😊😢😐", "Emojis (will be treated as unknown)"),
        ("A" * 1000, "Very long text"),
        ("UPPERCASE TEXT IS GOOD", "All caps"),
        ("mixed Case Text is FINE", "Mixed case"),
    ]
    
    print(f"Testing {len(edge_cases)} edge cases:\n")
    
    for i, (text, description) in enumerate(edge_cases, 1):
        try:
            result = analyzer.analyze_sentiment(text)
            status = "✅ HANDLED"
            details = f"{result.label.value} ({result.confidence:.2f})"
        except Exception as e:
            status = "❌ ERROR"
            details = str(e)
        
        print(f"{i:2d}. {status} - {description}")
        print(f"    Input: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"    Result: {details}\n")

def performance_test():
    """Performance testing"""
    print("🏃 Performance Test")
    print("-" * 40)
    
    analyzer = StandaloneSentimentAnalyzer()
    
    # Test different text lengths
    test_scenarios = [
        ("Short texts", ["Good", "Bad", "Ok"] * 100),
        ("Medium texts", ["This is a good product with nice features"] * 100),
        ("Long texts", ["This is a very long review with many words describing the product in detail and explaining why it's good or bad with lots of context and examples"] * 50),
    ]
    
    for scenario_name, texts in test_scenarios:
        print(f"\n📈 {scenario_name} ({len(texts)} texts):")
        
        start_time = time.time()
        results = analyzer.batch_analyze(texts)
        total_time = time.time() - start_time
        
        successful = len([r for r in results if r.confidence > 0])
        throughput = len(texts) / total_time
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        print(f"  Processing Time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} texts/second")
        print(f"  Success Rate: {successful}/{len(texts)} ({successful/len(texts)*100:.1f}%)")
        print(f"  Avg Confidence: {avg_confidence:.3f}")

def main():
    """Main demo runner"""
    print("🚀 Standalone Sentiment Analyzer Demo")
    print("=" * 50)
    print("Pure Python implementation with no external dependencies")
    print("=" * 50)
    
    # Run all demos
    demos = [
        ("Basic Functionality", demo_basic_functionality),
        ("Batch Processing", demo_batch_processing),
        ("Edge Cases & Robustness", demo_edge_cases),
        ("Performance Testing", performance_test),
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        
        try:
            start_time = time.time()
            demo_result = demo_func()
            duration = time.time() - start_time
            
            results[demo_name] = {
                "status": "SUCCESS",
                "duration": duration,
                "result": demo_result if demo_result else "Completed"
            }
            
            print(f"\n✅ {demo_name} completed in {duration:.2f}s")
            
        except Exception as e:
            results[demo_name] = {
                "status": "ERROR", 
                "duration": 0,
                "error": str(e)
            }
            print(f"\n❌ {demo_name} failed: {str(e)}")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Demo Summary")
    print(f"{'='*50}")
    
    successful_demos = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    total_demos = len(results)
    
    print(f"Completed: {successful_demos}/{total_demos} demos")
    print(f"Success Rate: {successful_demos/total_demos*100:.1f}%")
    
    for demo_name, result in results.items():
        status_emoji = "✅" if result["status"] == "SUCCESS" else "❌"
        print(f"  {status_emoji} {demo_name}: {result['status']}")
        if result.get("error"):
            print(f"      Error: {result['error']}")
    
    # Save results
    with open("standalone_demo_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Demo results saved to: standalone_demo_results.json")
    
    # Final analyzer stats
    print(f"\n📈 Final Statistics:")
    analyzer = StandaloneSentimentAnalyzer()
    test_text = "This is a final test to get stats"
    analyzer.analyze_sentiment(test_text)  # Generate some stats
    stats = analyzer.get_stats()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    if successful_demos == total_demos:
        print(f"\n🎉 All demos completed successfully!")
        return True
    else:
        print(f"\n⚠️  Some demos failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)