#!/usr/bin/env python3
"""
Simple Sentiment Analysis Core Implementation - Generation 1
No external dependencies for immediate execution
"""
import json
import logging
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4


class SentimentResult:
    """Sentiment analysis result"""
    def __init__(self, text: str, sentiment: str, confidence: float, 
                 scores: Dict[str, float], timestamp: datetime, 
                 model_used: str, processing_time: float):
        self.text = text
        self.sentiment = sentiment
        self.confidence = confidence
        self.scores = scores
        self.timestamp = timestamp
        self.model_used = model_used
        self.processing_time = processing_time
    
    def to_dict(self):
        return {
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': self.confidence,
            'scores': self.scores,
            'timestamp': self.timestamp.isoformat(),
            'model_used': self.model_used,
            'processing_time': self.processing_time
        }


class SentimentBatch:
    """Batch sentiment analysis results"""
    def __init__(self, results: List[SentimentResult], total_processed: int,
                 batch_id: str, created_at: datetime, processing_time: float):
        self.results = results
        self.total_processed = total_processed
        self.batch_id = batch_id
        self.created_at = created_at
        self.processing_time = processing_time


class SentimentAnalyzer:
    """Core sentiment analysis engine - Generation 1: Simple & Working"""
    
    def __init__(self, model_name: str = "rule_based_v1"):
        self.model_name = model_name
        self.logger = self._setup_logging()
        
        # Enhanced lexicons for better accuracy
        self.positive_words = {
            # Basic positive
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'awesome',
            'brilliant', 'perfect', 'outstanding', 'superb', 'magnificent',
            'delighted', 'thrilled', 'excited', 'beautiful', 'incredible',
            # Extended positive
            'best', 'better', 'superior', 'phenomenal', 'remarkable', 'impressive',
            'valuable', 'useful', 'helpful', 'beneficial', 'positive', 'success',
            'successful', 'win', 'winner', 'victory', 'triumph', 'accomplish',
            'achieve', 'effective', 'efficient', 'smooth', 'easy', 'simple',
            'comfortable', 'convenient', 'nice', 'pleasant', 'attractive',
            'pretty', 'gorgeous', 'stunning', 'breathtaking', 'marvelous'
        }
        
        self.negative_words = {
            # Basic negative
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad',
            'angry', 'frustrated', 'disappointed', 'annoying', 'disgusting',
            'pathetic', 'useless', 'worthless', 'stupid', 'ridiculous',
            'outrageous', 'unacceptable', 'failed', 'broken', 'wrong',
            # Extended negative
            'worst', 'worse', 'inferior', 'poor', 'low', 'weak', 'fail',
            'failure', 'lose', 'loss', 'defeat', 'problem', 'issue', 'trouble',
            'difficult', 'hard', 'complex', 'complicated', 'confusing',
            'uncomfortable', 'inconvenient', 'ugly', 'disgusting', 'nasty',
            'gross', 'boring', 'dull', 'slow', 'expensive', 'costly', 'waste',
            'regret', 'sorry', 'mistake', 'error', 'damage', 'harm', 'hurt'
        }
        
        # Intensifiers and modifiers
        self.intensifiers = {
            'very': 1.5, 'really': 1.4, 'extremely': 1.8, 'incredibly': 1.7,
            'absolutely': 1.6, 'totally': 1.5, 'completely': 1.6, 'quite': 1.3,
            'rather': 1.2, 'fairly': 1.2, 'pretty': 1.3, 'highly': 1.5,
            'tremendously': 1.8, 'exceptionally': 1.7, 'remarkably': 1.6,
            'significantly': 1.5, 'substantially': 1.4, 'considerably': 1.4
        }
        
        self.diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'barely': 0.5, 'hardly': 0.4,
            'scarcely': 0.4, 'little': 0.6, 'bit': 0.7, 'kind': 0.8,
            'sort': 0.8, 'almost': 0.9, 'nearly': 0.9, 'mostly': 0.9
        }
        
        self.negation_words = {
            'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither',
            'nor', 'nobody', 'cant', "can't", 'cannot', 'wont', "won't",
            'wouldnt', "wouldn't", 'shouldnt', "shouldn't", 'dont', "don't",
            'doesnt', "doesn't", 'didnt', "didn't", 'isnt', "isn't",
            'arent', "aren't", 'wasnt', "wasn't", 'werent', "weren't"
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger(f'sentiment_analyzer_{self.model_name}')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Handle contractions
        contractions = {
            "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "i'd": "i would", "you'd": "you would",
            "he'd": "he would", "she'd": "she would", "we'd": "we would",
            "they'd": "they would", "can't": "cannot", "won't": "will not",
            "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "isn't": "is not", "aren't": "are not", "wasn't": "was not",
            "weren't": "were not", "haven't": "have not", "hasn't": "has not",
            "hadn't": "had not", "mustn't": "must not", "needn't": "need not"
        }
        
        for contraction, expanded in contractions.items():
            text = text.replace(contraction, expanded)
            
        return text
        
    def analyze_text(self, text: str, prompt_variant: Optional[str] = None) -> SentimentResult:
        """
        Analyze sentiment of single text
        
        Args:
            text: Input text to analyze
            prompt_variant: Optional prompt to use (for evolutionary testing)
            
        Returns:
            SentimentResult with analysis
        """
        start_time = time.time()
        
        try:
            # Preprocess text
            clean_text = self._preprocess_text(text)
            
            # Calculate sentiment scores
            sentiment_scores = self._advanced_rule_sentiment(clean_text)
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])[0]
            confidence = sentiment_scores[primary_sentiment]
            
            processing_time = time.time() - start_time
            
            result = SentimentResult(
                text=text,
                sentiment=primary_sentiment,
                confidence=confidence,
                scores=sentiment_scores,
                timestamp=datetime.now(),
                model_used=self.model_name,
                processing_time=processing_time
            )
            
            self.logger.info(f"Analyzed: '{text[:30]}...' -> {primary_sentiment} ({confidence:.3f})")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            # Return neutral sentiment on error
            return SentimentResult(
                text=text,
                sentiment="neutral",
                confidence=0.5,
                scores={"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                timestamp=datetime.now(),
                model_used=self.model_name,
                processing_time=time.time() - start_time
            )
    
    def _advanced_rule_sentiment(self, text: str) -> Dict[str, float]:
        """Advanced rule-based sentiment analysis with context awareness"""
        
        words = text.split()
        if not words:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        
        positive_score = 0.0
        negative_score = 0.0
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Current modifiers
            intensity = 1.0
            negation = False
            
            # Look back for modifiers (up to 3 words)
            lookback_start = max(0, i - 3)
            context = words[lookback_start:i]
            
            # Check for intensifiers and diminishers
            for ctx_word in context:
                if ctx_word in self.intensifiers:
                    intensity = max(intensity, self.intensifiers[ctx_word])
                elif ctx_word in self.diminishers:
                    intensity = min(intensity, self.diminishers[ctx_word])
                    
            # Check for negation (within 2 words)
            negation_context = words[max(0, i-2):i]
            for neg_word in negation_context:
                if neg_word in self.negation_words:
                    negation = True
                    break
            
            # Score the word
            base_score = 1.0
            final_score = base_score * intensity
            
            if word in self.positive_words:
                if negation:
                    negative_score += final_score
                else:
                    positive_score += final_score
                    
            elif word in self.negative_words:
                if negation:
                    positive_score += final_score
                else:
                    negative_score += final_score
                    
            i += 1
        
        # Handle special patterns
        text_lower = text.lower()
        
        # Boost for exclamation marks (excitement)
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            if positive_score > negative_score:
                positive_score += exclamation_count * 0.3
            elif negative_score > positive_score:
                negative_score += exclamation_count * 0.3
        
        # Boost for all caps words (intensity)
        caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
        caps_boost = len(caps_words) * 0.2
        if positive_score > negative_score:
            positive_score += caps_boost
        elif negative_score > positive_score:
            negative_score += caps_boost
        
        # Question marks often indicate uncertainty (neutral bias)
        question_count = text.count('?')
        neutral_bias = question_count * 0.1
        
        # Calculate final scores
        total_sentiment = positive_score + negative_score
        
        if total_sentiment == 0:
            # No sentiment words found - neutral with slight variations based on length
            base_neutral = 0.6 + neutral_bias
            remaining = (1.0 - base_neutral) / 2
            return {
                "positive": remaining,
                "negative": remaining,
                "neutral": base_neutral
            }
        
        # Normalize sentiment scores
        pos_ratio = positive_score / total_sentiment
        neg_ratio = negative_score / total_sentiment
        
        # Calculate confidence based on score difference
        score_diff = abs(pos_ratio - neg_ratio)
        
        # If scores are very close, lean towards neutral
        if score_diff < 0.2:
            neutral_strength = 0.4 + (0.2 - score_diff) + neutral_bias
            remaining = (1.0 - neutral_strength) / 2
            
            if pos_ratio > neg_ratio:
                return {
                    "positive": remaining + score_diff/4,
                    "negative": remaining - score_diff/4,
                    "neutral": neutral_strength
                }
            else:
                return {
                    "positive": remaining - score_diff/4,
                    "negative": remaining + score_diff/4,
                    "neutral": neutral_strength
                }
        
        # Clear sentiment winner
        if pos_ratio > neg_ratio:
            confidence = min(0.95, 0.5 + pos_ratio)
            neutral_score = max(0.05, 0.3 - score_diff + neutral_bias)
            negative_score = max(0.05, 1.0 - confidence - neutral_score)
            return {
                "positive": confidence,
                "negative": negative_score,
                "neutral": neutral_score
            }
        else:
            confidence = min(0.95, 0.5 + neg_ratio)
            neutral_score = max(0.05, 0.3 - score_diff + neutral_bias)
            positive_score = max(0.05, 1.0 - confidence - neutral_score)
            return {
                "positive": positive_score,
                "negative": confidence,
                "neutral": neutral_score
            }
    
    def analyze_batch(self, texts: List[str]) -> SentimentBatch:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            SentimentBatch with all results
        """
        start_time = time.time()
        batch_id = str(uuid4())[:8]
        
        self.logger.info(f"Processing batch {batch_id} with {len(texts)} texts")
        
        results = []
        for i, text in enumerate(texts, 1):
            if i % 10 == 0 or i == len(texts):
                print(f"Processing {i}/{len(texts)}...")
            result = self.analyze_text(text)
            results.append(result)
            
        processing_time = time.time() - start_time
        
        batch = SentimentBatch(
            results=results,
            total_processed=len(results),
            batch_id=batch_id,
            created_at=datetime.now(),
            processing_time=processing_time
        )
        
        self.logger.info(f"Batch {batch_id} completed in {processing_time:.2f}s")
        return batch
    
    def get_sentiment_distribution(self, results: List[SentimentResult]) -> Dict[str, float]:
        """Calculate sentiment distribution from results"""
        if not results:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            
        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for result in results:
            counts[result.sentiment] += 1
            
        total = len(results)
        return {k: round(v / total, 3) for k, v in counts.items()}
    
    def get_confidence_stats(self, results: List[SentimentResult]) -> Dict[str, float]:
        """Calculate confidence statistics"""
        if not results:
            return {}
            
        confidences = [r.confidence for r in results]
        return {
            "mean_confidence": round(sum(confidences) / len(confidences), 3),
            "min_confidence": round(min(confidences), 3),
            "max_confidence": round(max(confidences), 3),
            "high_confidence_ratio": round(sum(1 for c in confidences if c > 0.8) / len(confidences), 3)
        }
    
    def export_results(self, batch: SentimentBatch, filepath: str) -> None:
        """Export batch results to JSON file"""
        output_data = {
            "batch_info": {
                "batch_id": batch.batch_id,
                "total_processed": batch.total_processed,
                "created_at": batch.created_at.isoformat(),
                "processing_time": batch.processing_time,
                "avg_processing_per_text": round(batch.processing_time / batch.total_processed, 6),
                "model_used": self.model_name
            },
            "results": [result.to_dict() for result in batch.results],
            "analytics": {
                "sentiment_distribution": self.get_sentiment_distribution(batch.results),
                "confidence_stats": self.get_confidence_stats(batch.results)
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Results exported to {filepath}")


def main():
    """Demo Generation 1 functionality"""
    print("🎯 SENTIMENT ANALYZER - GENERATION 1: MAKE IT WORK")
    print("=" * 60)
    
    analyzer = SentimentAnalyzer()
    
    # Comprehensive test data covering edge cases
    test_texts = [
        "I absolutely love this product! It's incredibly amazing and works perfectly every time.",
        "This is absolutely terrible. Worst purchase ever. Complete waste of money and time.",
        "It's okay, nothing special but it does the job adequately.",
        "Not bad, could be better but I'm quite satisfied overall with the quality.",
        "Outstanding quality! Really exceeded my expectations significantly.",
        "Awful experience. Very disappointed and extremely frustrated with the service.",
        "Pretty good value for money. Happy with the purchase and would recommend.",
        "Meh. It's fine I guess. Nothing to write home about.",
        "Exceptional service! Will definitely recommend to others without hesitation.",
        "This product is completely broken and totally useless. Don't buy it!",
        # Edge cases
        "I don't hate it, but I don't love it either.",
        "NOT good, but not terrible.",
        "This isn't bad at all! Actually quite good.",
        "I can't say I'm not satisfied with this purchase.",
        "???",
        "WOW! AMAZING!!!! BEST EVER!!!!",
        "why is this so complicated? nothing works properly...",
        "Excellent! No complaints whatsoever. Highly recommended!",
        "Could be worse, I suppose. At least it arrived on time.",
        "Surprisingly decent for the price. Better than expected."
    ]
    
    print(f"Analyzing {len(test_texts)} test cases...")
    
    # Analyze batch
    batch_result = analyzer.analyze_batch(test_texts)
    
    print(f"\n📊 DETAILED RESULTS")
    print("-" * 60)
    
    # Display results
    for i, result in enumerate(batch_result.results, 1):
        print(f"\n{i:2d}. Text: {result.text}")
        print(f"    Sentiment: {result.sentiment.upper()} (confidence: {result.confidence:.3f})")
        print(f"    Detailed scores: {result.scores}")
        print(f"    Processing time: {result.processing_time*1000:.2f}ms")
    
    # Summary statistics
    distribution = analyzer.get_sentiment_distribution(batch_result.results)
    confidence_stats = analyzer.get_confidence_stats(batch_result.results)
    
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Total processed: {batch_result.total_processed}")
    print(f"Total processing time: {batch_result.processing_time:.3f}s")
    print(f"Average per text: {batch_result.processing_time*1000/batch_result.total_processed:.2f}ms")
    print(f"Throughput: {batch_result.total_processed/batch_result.processing_time:.1f} texts/second")
    
    print(f"\n📊 SENTIMENT DISTRIBUTION")
    for sentiment, ratio in distribution.items():
        print(f"{sentiment.capitalize()}: {ratio:.1%} ({int(ratio * batch_result.total_processed)} texts)")
    
    print(f"\n🎯 CONFIDENCE METRICS")
    for metric, value in confidence_stats.items():
        print(f"{metric.replace('_', ' ').title()}: {value}")
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"/root/repo/sentiment_results_{timestamp}.json"
    analyzer.export_results(batch_result, output_file)
    print(f"\n💾 Results exported to: {output_file}")
    
    # Generation 1 Success Validation
    print(f"\n✅ GENERATION 1 VALIDATION")
    print("=" * 60)
    print("✓ Core sentiment analysis functionality implemented")
    print("✓ Rule-based algorithm with lexicon matching")  
    print("✓ Batch processing capability")
    print("✓ Confidence scoring system")
    print("✓ Context-aware negation handling")
    print("✓ Intensifier and diminisher recognition")
    print("✓ JSON export functionality")
    print("✓ Comprehensive logging")
    print("✓ Error handling and graceful degradation")
    print("✓ Performance metrics and analytics")
    
    return batch_result


if __name__ == "__main__":
    main()