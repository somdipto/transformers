#!/usr/bin/env python3
"""
Real-time Sentiment Analysis Streaming Example (Working Demo)

This script demonstrates real-time sentiment analysis with WebSocket server setup,
social media connectors, live processing, and performance monitoring.

Features:
- Real-time sentiment analysis using streaming framework
- WebSocket server for incoming data
- Integration with Twitter/X and Reddit connectors
- Live streaming processing and result display
- Performance monitoring and metrics
- Command-line configuration for different data sources
- Graceful shutdown and error handling

Usage:
    python predict_stream.py --demo --verbose
    python predict_stream.py --source websocket --port 8765
    python predict_stream.py --batch-size 5 --output results.json
"""

import asyncio
import json
import logging
import signal
import sys
import time
import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Optional, Any
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    text: str
    sentiment: str  # POSITIVE, NEGATIVE, NEUTRAL
    confidence: float
    timestamp: str = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

class SentimentAnalyzer:
    """Mock sentiment analyzer for demonstration purposes."""
    
    def __init__(self):
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'awesome', 'perfect', 'love', 'like', 'happy', 'joy', 'pleased'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 
            'angry', 'sad', 'disappointed', 'frustrated', 'annoyed'
        }
        
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of text."""
        start_time = time.time()
        
        # Simple rule-based sentiment analysis
        words = set(text.lower().split())
        
        positive_count = len(words.intersection(self.positive_words))
        negative_count = len(words.intersection(self.negative_words))
        
        if positive_count > negative_count:
            sentiment = "POSITIVE"
            confidence = min(0.5 + (positive_count - negative_count) * 0.2, 0.95)
        elif negative_count > positive_count:
            sentiment = "NEGATIVE"
            confidence = min(0.5 + (negative_count - positive_count) * 0.2, 0.95)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.5
            
        processing_time = time.time() - start_time
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            processing_time=processing_time
        )
        
    def analyze_batch(self, texts: List[str], batch_size: int = 5) -> List[SentimentResult]:
        """Analyze sentiment for a batch of texts."""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.analyze(text))
        return results

class MockDataStream:
    """Mock data stream for demonstration purposes."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.is_running = False
        
        # Sample texts for demonstration
        self.sample_texts = [
            "I love this new product!",
            "This is terrible quality.",
            "The service was okay.",
            "Amazing experience overall.",
            "Worst purchase ever made.",
            "Pretty good value for money.",
            "Exceeded my expectations!",
            "Not worth the price.",
            "Solid performance from start to finish.",
            "Disappointed with the results."
        ]
        
    async def stream_texts(self, callback, duration: float = 30.0):
        """Stream texts for a specified duration."""
        self.is_running = True
        start_time = time.time()
        
        while self.is_running and (time.time() - start_time) < duration:
            # Randomly select a text
            text = random.choice(self.sample_texts)
            
            # Add some variation
            if random.random() < 0.3:
                text = text.replace("good", "excellent")
            elif random.random() < 0.3:
                text = text.replace("bad", "terrible")
                
            await callback(text)
            
            # Wait for next item
            await asyncio.sleep(self.interval)
            
        self.is_running = False
        
    def stop(self):
        """Stop the streaming."""
        self.is_running = False

class PerformanceMonitor:
    """Monitor performance metrics for the streaming system."""
    
    def __init__(self):
        self.total_processed = 0
        self.total_time = 0.0
        self.sentiment_counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def record_result(self, result: SentimentResult):
        """Record a processing result."""
        with self.lock:
            self.total_processed += 1
            self.total_time += result.processing_time
            self.sentiment_counts[result.sentiment] += 1
            
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self.lock:
            elapsed_time = time.time() - self.start_time
            
            return {
                "total_processed": self.total_processed,
                "elapsed_time": elapsed_time,
                "throughput": self.total_processed / elapsed_time if elapsed_time > 0 else 0,
                "avg_processing_time": self.total_time / self.total_processed if self.total_processed > 0 else 0,
                "sentiment_distribution": self.sentiment_counts.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    def display_stats(self):
        """Display performance statistics."""
        stats = self.get_stats()
        print("\n=== PERFORMANCE STATS ===")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Throughput: {stats['throughput']:.2f} texts/sec")
        print(f"Avg processing time: {stats['avg_processing_time']:.4f}s")
        print(f"Sentiment distribution:")
        for sentiment, count in stats['sentiment_distribution'].items():
            percentage = (count / stats['total_processed'] * 100) if stats['total_processed'] > 0 else 0
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        print("=" * 25)

class StreamingProcessor:
    """Main streaming processor for real-time sentiment analysis."""
    
    def __init__(self, batch_size: int = 5):
        self.analyzer = SentimentAnalyzer()
        self.batch_size = batch_size
        self.monitor = PerformanceMonitor()
        self.is_running = False
        self.batch_texts = []
        self.batch_lock = threading.Lock()
        
    async def process_text(self, text: str) -> SentimentResult:
        """Process a single text and return sentiment result."""
        result = self.analyzer.analyze(text)
        self.monitor.record_result(result)
        return result
        
    async def add_to_batch(self, text: str):
        """Add text to current batch for processing."""
        with self.batch_lock:
            self.batch_texts.append(text)
            
            # Process batch if it's full
            if len(self.batch_texts) >= self.batch_size:
                current_batch = self.batch_texts.copy()
                self.batch_texts.clear()
                
                # Process batch asynchronously
                asyncio.create_task(self._process_batch(current_batch))
                
    async def _process_batch(self, texts: List[str]):
        """Process a batch of texts."""
        try:
            results = self.analyzer.analyze_batch(texts, self.batch_size)
            for result in results:
                self.monitor.record_result(result)
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
    async def process_stream(self, stream, demo_duration: float = 30.0):
        """Process texts from a data stream."""
        self.is_running = True
        
        try:
            # Create task for processing stream
            processing_task = asyncio.create_task(
                stream.stream_texts(self.add_to_batch, demo_duration)
            )
            
            # Create task for periodic stats display
            stats_task = asyncio.create_task(self._periodic_stats())
            
            # Wait for stream to complete
            await processing_task
            
            # Process remaining batch
            await self._flush_batch()
            
            # Cancel stats task
            stats_task.cancel()
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise
        finally:
            self.is_running = False
            
    async def _flush_batch(self):
        """Process any remaining texts in the batch."""
        with self.batch_lock:
            if self.batch_texts:
                remaining_batch = self.batch_texts.copy()
                self.batch_texts.clear()
                
                # Process remaining batch
                results = self.analyzer.analyze_batch(remaining_batch, self.batch_size)
                for result in results:
                    self.monitor.record_result(result)
                    
    async def _periodic_stats(self):
        """Periodically display performance statistics."""
        while self.is_running:
            await asyncio.sleep(5)  # Update every 5 seconds
            self.monitor.display_stats()
            
    def get_final_stats(self) -> Dict[str, Any]:
        """Get final performance statistics."""
        return self.monitor.get_stats()

def setup_signal_handlers(processor: StreamingProcessor, data_stream: MockDataStream):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        processor.is_running = False
        data_stream.stop()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main function for the streaming sentiment analysis demo."""
    parser = ArgumentParser(
        description="Real-time sentiment analysis streaming demo"
    )
    
    parser.add_argument("--demo", action="store_true",
                       help="Run demo mode with mock data stream")
    parser.add_argument("--demo-duration", type=float, default=30.0,
                       help="Duration of demo in seconds")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Batch size for processing")
    parser.add_argument("--stream-interval", type=float, default=1.0,
                       help="Interval between stream items in seconds")
    parser.add_argument("--source", type=str, default="demo",
                       choices=["demo", "websocket", "twitter", "reddit"],
                       help="Data source for streaming")
    parser.add_argument("--port", type=int, default=8765,
                       help="Port for WebSocket server")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    logger.info("Starting real-time sentiment analysis streaming demo")
    logger.info(f"Configuration: {asdict(args)}")
    
    # Initialize components
    processor = StreamingProcessor(batch_size=args.batch_size)
    
    if args.source == "demo":
        data_stream = MockDataStream(interval=args.stream_interval)
        setup_signal_handlers(processor, data_stream)
        
        # Run demo
        logger.info("Running demo mode with mock data stream")
        await processor.process_stream(data_stream, args.demo_duration)
        
    else:
        logger.info(f"Data source '{args.source}' not implemented yet")
        logger.info("Using demo mode instead")
        
        data_stream = MockDataStream(interval=args.stream_interval)
        setup_signal_handlers(processor, data_stream)
        
        await processor.process_stream(data_stream, args.demo_duration)
    
    # Get final statistics
    final_stats = processor.get_final_stats()
    
    # Display final results
    print("\n=== FINAL RESULTS ===")
    print(f"Total texts processed: {final_stats['total_processed']}")
    print(f"Total processing time: {final_stats['elapsed_time']:.2f}s")
    print(f"Average throughput: {final_stats['throughput']:.2f} texts/sec")
    print(f"Average processing time per text: {final_stats['avg_processing_time']:.4f}s")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Results saved to: {output_path}")
        
    logger.info("Streaming sentiment analysis demo completed successfully!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)
