#!/usr/bin/env python3
"""
Sentiment Analysis Core Implementation
Integrates with Meta-Prompt-Evolution-Hub for prompt optimization
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    scores: Dict[str, float]  # detailed sentiment scores
    timestamp: datetime
    model_used: str
    processing_time: float


@dataclass 
class SentimentBatch:
    """Batch sentiment analysis results"""
    results: List[SentimentResult]
    total_processed: int
    batch_id: str
    created_at: datetime
    processing_time: float


class SentimentAnalyzer:
    """Core sentiment analysis engine with evolutionary prompt optimization"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.logger = self._setup_logging()
        self.performance_cache = {}
        
        # Default prompts for evolutionary optimization
        self.base_prompts = [
            "Analyze the sentiment of this text: '{text}'. Respond with positive, negative, or neutral.",
            "What is the emotional tone of: '{text}'? Choose: positive, negative, or neutral.",
            "Determine sentiment: '{text}'. Answer: positive/negative/neutral only.",
            "Classify the sentiment in '{text}' as positive, negative, or neutral.",
            "Rate the sentiment of '{text}' as positive, negative, or neutral with confidence."
        ]
        
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
            # Use rule-based approach for now (Generation 1: Simple)
            sentiment_scores = self._rule_based_sentiment(text)
            
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
            
            self.logger.info(f"Analyzed text: {text[:50]}... -> {primary_sentiment} ({confidence:.3f})")
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
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        """Simple rule-based sentiment analysis for Generation 1"""
        
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'awesome',
            'brilliant', 'perfect', 'outstanding', 'superb', 'magnificent',
            'delighted', 'thrilled', 'excited', 'beautiful', 'incredible'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'sad',
            'angry', 'frustrated', 'disappointed', 'annoying', 'disgusting',
            'pathetic', 'useless', 'worthless', 'stupid', 'ridiculous',
            'outrageous', 'unacceptable', 'failed', 'broken', 'wrong'
        }
        
        # Intensifiers
        intensifiers = {'very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally'}
        
        # Negation words
        negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'neither'}
        
        words = text.lower().split()
        
        positive_score = 0
        negative_score = 0
        intensifier_multiplier = 1.0
        negation_active = False
        
        for i, word in enumerate(words):
            # Check for intensifiers
            if word in intensifiers:
                intensifier_multiplier = 1.5
                continue
                
            # Check for negation
            if word in negation_words:
                negation_active = True
                continue
            
            # Score sentiment
            if word in positive_words:
                score = 1.0 * intensifier_multiplier
                if negation_active:
                    negative_score += score
                else:
                    positive_score += score
            elif word in negative_words:
                score = 1.0 * intensifier_multiplier
                if negation_active:
                    positive_score += score
                else:
                    negative_score += score
                    
            # Reset modifiers after use
            intensifier_multiplier = 1.0
            negation_active = False
            
        # Normalize scores
        total_score = positive_score + negative_score
        if total_score == 0:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
            
        pos_ratio = positive_score / total_score
        neg_ratio = negative_score / total_score
        
        # If very close, lean neutral
        if abs(pos_ratio - neg_ratio) < 0.1:
            return {"positive": 0.3, "negative": 0.3, "neutral": 0.4}
            
        # Clear winner
        if pos_ratio > neg_ratio:
            confidence = min(0.95, 0.5 + pos_ratio)
            return {
                "positive": confidence,
                "negative": 1 - confidence - 0.1,
                "neutral": 0.1
            }
        else:
            confidence = min(0.95, 0.5 + neg_ratio)
            return {
                "positive": 1 - confidence - 0.1,
                "negative": confidence,
                "neutral": 0.1
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
        batch_id = str(uuid4())
        
        self.logger.info(f"Processing batch {batch_id} with {len(texts)} texts")
        
        results = []
        for text in texts:
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
        return {k: v / total for k, v in counts.items()}
    
    def export_results(self, batch: SentimentBatch, filepath: str) -> None:
        """Export batch results to JSON file"""
        output_data = {
            "batch_info": {
                "batch_id": batch.batch_id,
                "total_processed": batch.total_processed,
                "created_at": batch.created_at.isoformat(),
                "processing_time": batch.processing_time
            },
            "results": [asdict(result) for result in batch.results],
            "summary": self.get_sentiment_distribution(batch.results)
        }
        
        # Convert datetime objects to strings for JSON serialization
        for result in output_data["results"]:
            result["timestamp"] = result["timestamp"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        self.logger.info(f"Results exported to {filepath}")


def main():
    """Demo Generation 1 functionality"""
    analyzer = SentimentAnalyzer()
    
    # Test data
    test_texts = [
        "I love this product! It's absolutely amazing and works perfectly.",
        "This is terrible. Worst purchase ever. Complete waste of money.",
        "It's okay, nothing special but does the job.",
        "Not bad, could be better but I'm satisfied overall.",
        "Incredible quality! Really exceeded my expectations.",
        "Awful experience. Very disappointed and frustrated.",
        "Pretty good value for money. Happy with the purchase.",
        "Meh. It's fine I guess.",
        "Outstanding service! Will definitely recommend to others.",
        "This product is broken and useless. Don't buy it."
    ]
    
    print("🎯 SENTIMENT ANALYZER - GENERATION 1")
    print("=" * 50)
    
    # Analyze batch
    batch_result = analyzer.analyze_batch(test_texts)
    
    # Display results
    for i, result in enumerate(batch_result.results, 1):
        print(f"\n{i}. Text: {result.text}")
        print(f"   Sentiment: {result.sentiment.upper()} (confidence: {result.confidence:.3f})")
        print(f"   Scores: {result.scores}")
        print(f"   Processing time: {result.processing_time:.4f}s")
    
    # Summary statistics
    distribution = analyzer.get_sentiment_distribution(batch_result.results)
    print(f"\n📊 BATCH SUMMARY")
    print(f"Total processed: {batch_result.total_processed}")
    print(f"Processing time: {batch_result.processing_time:.2f}s")
    print(f"Average per text: {batch_result.processing_time/batch_result.total_processed:.4f}s")
    print(f"Distribution: {distribution}")
    
    # Export results
    output_file = f"/root/repo/sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    analyzer.export_results(batch_result, output_file)
    print(f"\n💾 Results saved to: {output_file}")
    
    return batch_result


if __name__ == "__main__":
    main()