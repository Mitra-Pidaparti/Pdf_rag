import os
import re
import pandas as pd
import numpy as np
import logging
import time
from difflib import SequenceMatcher
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional imports with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    util = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChunkComparator:
    """
    A comprehensive system for comparing text chunks between benchmark and system datasets
    using multiple similarity methods including OpenAI GPT-4o-mini-based semantic analysis with adaptive scoring.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize the comparator with OpenAI GPT-4o-mini client and similarity tools.
        
        Args:
            api_key: OpenAI API key
        """
        self.api_key = api_key
        self.model_name = "gpt-4o-mini"
        self.provider = "openai"
        self.client = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            ngram_range=(1, 2),
            max_features=10000
        )
        
        # Initialize sentence transformer model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer model: {e}")
        
        # Initialize OpenAI client
        self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize the OpenAI client."""
        if not self.api_key:
            logger.warning("No OpenAI API key provided - LLM semantic analysis will be disabled")
            return
        
        try:
            if OPENAI_AVAILABLE:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI GPT-4o-mini client initialized successfully")
            else:
                logger.warning("OpenAI library not available")
                
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def load_excel_files(self, benchmark_path: str, system_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load Excel files and return lists of chunk dictionaries.
        
        Args:
            benchmark_path: Path to benchmark Excel file
            system_path: Path to system Excel file
            
        Returns:
            Tuple of (benchmark_chunks, system_chunks)
        """
        try:
            benchmark_df = pd.read_excel(benchmark_path)
            system_df = pd.read_excel(system_path)
        except Exception as e:
            logger.error(f"Failed to load Excel files: {e}")
            raise

        # Extract valid chunks from benchmark
        benchmark_chunks = []
        for idx, row in benchmark_df.iterrows():
            content = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if content and content.lower() != 'nan':
                benchmark_chunks.append({
                    'index': idx,
                    'content': content,
                    'cleaned': self.preprocess_text(content)
                })

        # Extract valid chunks from system
        system_chunks = []
        for idx, row in system_df.iterrows():
            content = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if content and content.lower() != 'nan':
                system_chunks.append({
                    'index': idx,
                    'content': content,
                    'cleaned': self.preprocess_text(content)
                })

        logger.info(f"Loaded {len(benchmark_chunks)} benchmark and {len(system_chunks)} system chunks")
        return benchmark_chunks, system_chunks

    @staticmethod
    def preprocess_text(text: Any) -> str:
        """Clean and normalize text for comparison."""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower().strip()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text

    def calculate_exact_coverage(self, benchmark_text: str, system_text: str) -> float:
        """Calculate exact word overlap percentage."""
        if not benchmark_text or not system_text:
            return 0.0
        
        benchmark_words = set(benchmark_text.split())
        system_words = set(system_text.split())
        
        if not benchmark_words:
            return 0.0
        
        overlap = benchmark_words.intersection(system_words)
        return (len(overlap) / len(benchmark_words)) * 100

    def calculate_fuzzy_coverage(self, benchmark_text: str, system_text: str) -> float:
        """Calculate fuzzy similarity using sentence transformers or fallback method."""
        if not benchmark_text or not system_text:
            return 0.0
        
        if self.embedding_model is not None:
            try:
                embeddings = self.embedding_model.encode([benchmark_text, system_text], convert_to_tensor=True)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                return max(0.0, min(similarity * 100, 100.0))
            except Exception as e:
                logger.warning(f"Sentence transformer error: {e}")
        
        # Fallback to basic string similarity
        return SequenceMatcher(None, benchmark_text, system_text).ratio() * 100

    def calculate_tfidf_coverage(self, benchmark_text: str, system_text: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        if not benchmark_text or not system_text:
            return 0.0
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([benchmark_text, system_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except Exception as e:
            logger.warning(f"TF-IDF calculation error: {e}")
            return 0.0

    def calculate_semantic_coverage(self, benchmark_text: str, system_text: str) -> float:
        """Calculate semantic coverage using OpenAI GPT-4o-mini."""
        if not self.client or not benchmark_text or not system_text:
            return 0.0
        
        # Truncate texts if too long
        max_chars = 4000
        if len(benchmark_text) > max_chars:
            benchmark_text = benchmark_text[:max_chars] + "..."
        if len(system_text) > max_chars:
            system_text = system_text[:max_chars] + "..."
        
        prompt = f"""Analyze how much of the benchmark content is semantically covered by the system content.

Benchmark content:
"{benchmark_text}"

System content:
"{system_text}"

Rate the percentage (0-100) of benchmark information that is present semantically in the system content.
Return only the number."""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            score_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            numbers = re.findall(r'\d+\.?\d*', score_text)
            if numbers:
                score = float(numbers[0])
                return min(100.0, max(0.0, score))
            return 0.0
            
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return 0.0

    def calculate_adaptive_coverage(self, benchmark_chunk: Dict, contributing_chunks: List[Dict]) -> float:
        """
        Calculate coverage using adaptive decision tree logic:
        1. No matches found → 0% coverage
        2. Single high-confidence match (≥80%) → Use that match score
        3. Multiple moderate matches → Aggregate content and re-analyze
        4. Multiple low matches → If sum is moderate/good, aggregate and re-analyze
        """
        if not contributing_chunks:
            return 0.0
        
        # Get top scores
        top_score = contributing_chunks[0]['max_coverage']
        num_chunks = len(contributing_chunks)
        
        # Decision Tree Logic
        
        # Case 1: Single high-confidence match (≥80%)
        if top_score >= 80.0:
            logger.debug(f"Single high-confidence match: {top_score}%")
            return top_score
        
        # Case 2: Multiple chunks available
        if num_chunks >= 2:
            # Calculate sum of top 3 chunks for assessment
            top_chunks = contributing_chunks[:3]
            sum_coverage = sum(chunk['max_coverage'] for chunk in top_chunks)
            
            # Case 2a: Multiple moderate matches OR multiple low matches with good sum
            if (top_score >= 40.0) or (sum_coverage >= 60.0):
                logger.debug(f"Multiple chunks aggregation: top={top_score}%, sum={sum_coverage}%")
                
                # Aggregate content and re-analyze with OpenAI if available
                if self.client:
                    combined_content = " ".join([
                        chunk['full_content'] for chunk in top_chunks
                    ])
                    
                    aggregated_score = self.calculate_semantic_coverage(
                        benchmark_chunk['cleaned'], 
                        combined_content
                    )
                    logger.debug(f"OpenAI aggregated score: {aggregated_score}%")
                    return aggregated_score
                
                # Fallback: Weighted coverage if no OpenAI
                else:
                    weights = [1.0, 0.7, 0.5]  # Diminishing weights to account for overlap
                    weighted_coverage = sum(
                        chunk['max_coverage'] * weights[i] 
                        for i, chunk in enumerate(top_chunks)
                    )
                    final_score = min(100.0, weighted_coverage)
                    logger.debug(f"Weighted aggregated score: {final_score}%")
                    return final_score
        
        # Case 3: Single low match or insufficient sum - use top score
        logger.debug(f"Single/insufficient coverage: {top_score}%")
        return top_score

    def analyze_benchmark_chunk(self, benchmark_chunk: Dict, system_chunks: List[Dict],
                               use_llm: bool = True, coverage_threshold: float = 15.0,
                               max_chunks_to_combine: int = 3) -> Dict[str, Any]:
        """
        Analyze coverage of a benchmark chunk against all system chunks using adaptive logic.
        
        Args:
            benchmark_chunk: Dictionary containing benchmark chunk data
            system_chunks: List of system chunk dictionaries
            use_llm: Whether to use OpenAI for semantic analysis
            coverage_threshold: Minimum coverage threshold to consider a match (default: 15.0%)
            max_chunks_to_combine: Maximum chunks to combine for aggregation
            
        Returns:
            Dictionary containing coverage analysis results
        """
        benchmark_content = benchmark_chunk['cleaned']
        if not benchmark_content:
            return {
                'total_coverage': 0.0,
                'contributing_chunks': [],
                'method_scores': {'exact': 0.0, 'fuzzy': 0.0, 'tfidf': 0.0, 'semantic': 0.0},
                'coverage_pattern': 'none'
            }
        
        contributing_chunks = []
        
        for system_chunk in system_chunks:
            system_content = system_chunk['cleaned']
            if not system_content:
                continue
            
            # Calculate coverage using different methods
            exact_coverage = self.calculate_exact_coverage(benchmark_content, system_content)
            fuzzy_coverage = self.calculate_fuzzy_coverage(benchmark_content, system_content)
            tfidf_coverage = self.calculate_tfidf_coverage(benchmark_content, system_content)
            
            # Use OpenAI only for promising candidates (≥15% threshold) to save API calls
            semantic_coverage = 0.0
            max_coverage = max(exact_coverage, fuzzy_coverage, tfidf_coverage)
            
            if use_llm and max_coverage > coverage_threshold and self.client:
                semantic_coverage = self.calculate_semantic_coverage(benchmark_content, system_content)
                max_coverage = max(max_coverage, semantic_coverage)
            
            # Track chunks with significant coverage (≥15% threshold)
            if max_coverage > coverage_threshold:
                contributing_chunks.append({
                    'system_index': system_chunk['index'],
                    'system_content': (system_chunk['content'][:150] + "...") 
                                    if len(system_chunk['content']) > 150 else system_chunk['content'],
                    'full_content': system_chunk['content'],  # Store full content for aggregation
                    'exact_coverage': round(exact_coverage, 2),
                    'fuzzy_coverage': round(fuzzy_coverage, 2),
                    'tfidf_coverage': round(tfidf_coverage, 2),
                    'semantic_coverage': round(semantic_coverage, 2),
                    'max_coverage': round(max_coverage, 2)
                })
        
        # Sort by coverage score
        contributing_chunks.sort(key=lambda x: x['max_coverage'], reverse=True)
        
        # Use adaptive coverage calculation
        total_coverage = self.calculate_adaptive_coverage(benchmark_chunk, contributing_chunks)
        
        # Determine coverage pattern
        coverage_pattern = self.determine_coverage_pattern(contributing_chunks)
        
        # Calculate method averages
        method_scores = {'exact': 0.0, 'fuzzy': 0.0, 'tfidf': 0.0, 'semantic': 0.0}
        if contributing_chunks:
            top_chunks = contributing_chunks[:5]  # Top 5 for averaging
            method_scores = {
                'exact': sum(c['exact_coverage'] for c in top_chunks) / len(top_chunks),
                'fuzzy': sum(c['fuzzy_coverage'] for c in top_chunks) / len(top_chunks),
                'tfidf': sum(c['tfidf_coverage'] for c in top_chunks) / len(top_chunks),
                'semantic': sum(c['semantic_coverage'] for c in top_chunks) / len(top_chunks)
            }
        
        return {
            'total_coverage': round(total_coverage, 2),
            'contributing_chunks': contributing_chunks[:10],  # Top 10 matches
            'method_scores': {k: round(v, 2) for k, v in method_scores.items()},
            'coverage_pattern': coverage_pattern
        }

    def determine_coverage_pattern(self, contributing_chunks: List[Dict]) -> str:
        """Determine the pattern of coverage distribution."""
        if not contributing_chunks:
            return "none"
        
        top_score = contributing_chunks[0]['max_coverage']
        num_chunks = len(contributing_chunks)
        
        if num_chunks == 1 or top_score >= 80:
            return "concentrated"
        elif num_chunks >= 2 and top_score < 60:
            return "distributed"
        else:
            return "mixed"

    def process_all_chunks(self, benchmark_chunks: List[Dict], system_chunks: List[Dict],
                          use_llm: bool = True, coverage_threshold: float = 15.0) -> pd.DataFrame:
        """Process all benchmark chunks and calculate coverage scores."""
        results = []
        
        logger.info(f"Processing {len(benchmark_chunks)} benchmark chunks with OpenAI GPT-4o-mini...")
        logger.info(f"Using coverage threshold: {coverage_threshold}% for API calls")
        
        with tqdm(total=len(benchmark_chunks), desc="Analyzing chunks") as pbar:
            for benchmark_chunk in benchmark_chunks:
                coverage_analysis = self.analyze_benchmark_chunk(
                    benchmark_chunk, system_chunks, use_llm, coverage_threshold
                )
                
                result = {
                    'benchmark_index': benchmark_chunk['index'],
                    'benchmark_content': (benchmark_chunk['content'][:200] + "...") 
                                       if len(benchmark_chunk['content']) > 200 else benchmark_chunk['content'],
                    'content_length': len(benchmark_chunk['content']),
                    'total_coverage': coverage_analysis['total_coverage'],
                    'coverage_pattern': coverage_analysis['coverage_pattern'],
                    'num_contributing_chunks': len(coverage_analysis['contributing_chunks']),
                    'exact_method_score': coverage_analysis['method_scores']['exact'],
                    'fuzzy_method_score': coverage_analysis['method_scores']['fuzzy'],
                    'tfidf_method_score': coverage_analysis['method_scores']['tfidf'],
                    'semantic_method_score': coverage_analysis['method_scores']['semantic'],
                    'top_contributing_chunks': coverage_analysis['contributing_chunks'][:5]
                }
                results.append(result)
                pbar.update(1)
        
        return pd.DataFrame(results)

    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        if results_df.empty:
            return {"error": "No results to analyze"}
        
        total_chunks = len(results_df)
        avg_coverage = results_df['total_coverage'].mean()
        
        # Coverage distribution
        excellent = len(results_df[results_df['total_coverage'] >= 90])
        good = len(results_df[(results_df['total_coverage'] >= 70) & (results_df['total_coverage'] < 90)])
        moderate = len(results_df[(results_df['total_coverage'] >= 50) & (results_df['total_coverage'] < 70)])
        poor = len(results_df[(results_df['total_coverage'] >= 30) & (results_df['total_coverage'] < 50)])
        very_poor = len(results_df[results_df['total_coverage'] < 30])
        no_coverage = len(results_df[results_df['total_coverage'] == 0])
        
        # Coverage pattern analysis
        pattern_counts = results_df['coverage_pattern'].value_counts().to_dict()
        
        # Method performance
        method_performance = {
            'exact_avg': round(results_df['exact_method_score'].mean(), 2),
            'fuzzy_avg': round(results_df['fuzzy_method_score'].mean(), 2),
            'tfidf_avg': round(results_df['tfidf_method_score'].mean(), 2),
            'semantic_avg': round(results_df['semantic_method_score'].mean(), 2)
        }
        
        # Best and worst covered chunks
        best_covered = results_df.nlargest(5, 'total_coverage')[
            ['benchmark_index', 'benchmark_content', 'total_coverage', 'coverage_pattern', 'num_contributing_chunks']
        ].to_dict('records')
        
        worst_covered = results_df.nsmallest(5, 'total_coverage')[
            ['benchmark_index', 'benchmark_content', 'total_coverage', 'coverage_pattern', 'num_contributing_chunks']
        ].to_dict('records')
        
        summary = {
            'total_benchmark_chunks_analyzed': total_chunks,
            'average_coverage': round(avg_coverage, 2),
            'coverage_distribution': {
                'excellent_90_plus': excellent,
                'good_70_89': good,
                'moderate_50_69': moderate,
                'poor_30_49': poor,
                'very_poor_below_30': very_poor
            },
            'coverage_patterns': pattern_counts,
            'method_performance': method_performance,
            'chunks_with_no_coverage': no_coverage,
            'best_covered': best_covered,
            'worst_covered': worst_covered,
            'api_info': {
                'model_used': self.model_name,
                'provider': self.provider,
                'threshold_for_api_calls': 15.0
            }
        }
        
        return summary

    def save_results(self, results_df: pd.DataFrame, output_path: str = "chunk_coverage_results.xlsx"):
        """Save results to Excel file."""
        try:
            # Create a copy without the complex nested data for Excel export
            export_df = results_df.drop(['top_contributing_chunks'], axis=1)
            export_df.to_excel(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary report."""
        print("\n" + "="*80)
        print("ENHANCED BENCHMARK CHUNK COVERAGE ANALYSIS REPORT")
        print("(Powered by OpenAI GPT-4o-mini)")
        print("="*80)
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total benchmark chunks analyzed: {summary['total_benchmark_chunks_analyzed']}")
        print(f"  Average coverage: {summary['average_coverage']:.2f}%")
        print(f"  Chunks with no coverage: {summary['chunks_with_no_coverage']}")
        print(f"  API threshold: {summary['api_info']['threshold_for_api_calls']}%")
        print(f"  Model used: {summary['api_info']['model_used']}")
        
        print(f"\n📈 COVERAGE DISTRIBUTION:")
        dist = summary['coverage_distribution']
        print(f"  🟢 Excellent (90%+): {dist['excellent_90_plus']} chunks")
        print(f"  🔵 Good (70-89%): {dist['good_70_89']} chunks")
        print(f"  🟡 Moderate (50-69%): {dist['moderate_50_69']} chunks")
        print(f"  🟠 Poor (30-49%): {dist['poor_30_49']} chunks")
        print(f"  🔴 Very Poor (<30%): {dist['very_poor_below_30']} chunks")
        
        print(f"\n🎯 COVERAGE PATTERNS:")
        patterns = summary['coverage_patterns']
        for pattern, count in patterns.items():
            print(f"  {pattern.title()}: {count} chunks")
        
        print(f"\n🔍 METHOD PERFORMANCE:")
        perf = summary['method_performance']
        print(f"  Exact matching: {perf['exact_avg']:.2f}%")
        print(f"  Fuzzy matching: {perf['fuzzy_avg']:.2f}%")
        print(f"  TF-IDF similarity: {perf['tfidf_avg']:.2f}%")
        print(f"  Semantic similarity (GPT-4o-mini): {perf['semantic_avg']:.2f}%")
        
        print(f"\n🏆 BEST COVERED CHUNKS:")
        for i, chunk in enumerate(summary['best_covered'][:3], 1):
            print(f"  {i}. Index {chunk['benchmark_index']}: {chunk['total_coverage']:.1f}% ({chunk['coverage_pattern']})")
            print(f"     Preview: {chunk['benchmark_content'][:100]}...")
        
        print(f"\n⚠️  WORST COVERED CHUNKS:")
        for i, chunk in enumerate(summary['worst_covered'][:3], 1):
            print(f"  {i}. Index {chunk['benchmark_index']}: {chunk['total_coverage']:.1f}% ({chunk['coverage_pattern']})")
            print(f"     Preview: {chunk['benchmark_content'][:100]}...")
        
        print("="*80)


def main():
    """Main execution function with OpenAI GPT-4o-mini configuration."""
    
    # Configuration
    BENCHMARK_FILE = "Compare_outputs/benchmark.xlsx"
    SYSTEM_FILE = "Compare_outputs/system.xlsx"
    OUTPUT_FILE = "chunk_coverage_results.xlsx"
    
    # OpenAI API Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Analysis settings
    USE_LLM = True  # Set to False for faster processing without OpenAI
    COVERAGE_THRESHOLD = 15.0  # Minimum coverage to consider API calls (increased from 5.0)
    
    try:
        # Initialize comparator with OpenAI GPT-4o-mini
        logger.info("Initializing chunk comparator with OpenAI GPT-4o-mini...")
        comparator = ChunkComparator(api_key=API_KEY)
        
        # Load data
        logger.info("Loading Excel files...")
        benchmark_chunks, system_chunks = comparator.load_excel_files(BENCHMARK_FILE, SYSTEM_FILE)
        
        # Process chunks
        logger.info("Processing chunks with adaptive scoring and 15% API threshold...")
        results_df = comparator.process_all_chunks(
            benchmark_chunks, 
            system_chunks, 
            use_llm=USE_LLM, 
            coverage_threshold=COVERAGE_THRESHOLD
        )
        
        # Generate and display summary
        logger.info("Generating enhanced summary report...")
        summary = comparator.generate_summary_report(results_df)
        comparator.print_summary(summary)
        
        # Save results
        comparator.save_results(results_df, OUTPUT_FILE)
        
        logger.info("Enhanced analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
