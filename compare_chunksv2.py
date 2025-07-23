import pandas as pd
import numpy as np
from openai import OpenAI
import re
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any
import time
from tqdm import tqdm
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util




load_dotenv
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelContentComparator:
    def __init__(self, openai_api_key: str,model_name = "gpt-4-turbo"):
        """
        Initialize the comparator with OpenAI client
        
        Args:
            openai_api_key: Your OpenAI API key
            model_name: OpenAI model to use for semantic analysis
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model_name = model_name
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def load_excel_files(self, benchmark_path: str, system_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Excel files and return DataFrames"""
        try:
            benchmark_df = pd.read_excel(benchmark_path)
            system_df = pd.read_excel(system_path)
            logger.info(f"Loaded benchmark: {benchmark_df.shape}, system: {system_df.shape}")
            return benchmark_df, system_df
        except Exception as e:
            logger.error(f"Error loading Excel files: {e}")
            raise
    
    def preprocess_text(self, text: Any) -> str:
        """Clean and preprocess text for comparison"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower().strip()
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
        return text
    
    def get_all_system_content(self, system_df: pd.DataFrame) -> str:
        """Combine all system content into a single string for searching"""
        all_content = []
        for col in system_df.columns:
            for value in system_df[col].dropna():
                cleaned = self.preprocess_text(value)
                if cleaned:
                    all_content.append(cleaned)
        return " ".join(all_content)
    


    def exact_substring_match(self, benchmark_text: str, system_content: str) -> float:
        """Calculate exact substring match percentage"""
        if not benchmark_text:
            return 0.0
        
        # Get embeddings
        embeddings = self.embedding_model.encode([benchmark_text, system_content], convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # Return as percentage
        return max(0.0, min(similarity * 100, 100.0))

    
    def fuzzy_match_aggressive(self, benchmark_text: str, system_content: str) -> float:
        """Aggressive fuzzy matching using sequence matcher"""
        if not benchmark_text:
            return 0.0
        
        # Split into sentences/phrases for better matching
        benchmark_phrases = [p.strip() for p in re.split(r'[.!?;]', benchmark_text) if p.strip()]
        
        total_similarity = 0
        for phrase in benchmark_phrases:
            # Find best match in system content using sliding window
            best_match = 0
            words = phrase.split()
            
            for i in range(len(system_content.split()) - len(words) + 1):
                window = " ".join(system_content.split()[i:i+len(words)])
                similarity = SequenceMatcher(None, phrase, window).ratio()
                best_match = max(best_match, similarity)
            
            total_similarity += best_match
        
        return (total_similarity / len(benchmark_phrases)) * 100 if benchmark_phrases else 0
    
    def tfidf_similarity(self, benchmark_text: str, system_content: str) -> float:
        """Calculate TF-IDF cosine similarity"""
        if not benchmark_text or not system_content:
            return 0.0
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([benchmark_text, system_content])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 0.0
    
    def semantic_similarity_openai(self, benchmark_text: str, system_content: str) -> float:
        """Use OpenAI to assess semantic similarity"""
        if not benchmark_text or not system_content:
            return 0.0
        
        # Truncate content if too long to avoid token limits
        max_chars = 40000
        if len(system_content) > max_chars:
            system_content = system_content[:max_chars] + "..."
        
        prompt = f"""
        Compare the following benchmark text with system content and rate how much of the benchmark information is reflected in the system content.
        
        Benchmark text: "{benchmark_text}"
        
        System content: "{system_content}"
        
        Score the system content on a scale from 0 to 100 based on how completely it includes the information in the benchmark:

        - 100 = All benchmark information is fully present
        - 80-99 = Most information is present with minor gaps
        - 60-79 = A significant portion is present, but some important parts are missing
        - 40-59 = Partial information is present, but much is missing
        - 20-39 = Minimal relevant information present
        - 0-19 = Little to none of the benchmark information is included

        Respond with a single number only — no explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(re.findall(r'\d+', score_text)[0])
            return min(100, max(0, score))
        
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return 0.0
    
    def analyze_cell_content(self, benchmark_text: str, system_df: pd.DataFrame, 
                           system_content: str, use_openai: bool = True) -> Dict[str, float]:
        """Analyze a single benchmark cell against system content"""
        cleaned_benchmark = self.preprocess_text(benchmark_text)
        
        if not cleaned_benchmark:
            return {
                'exact_match': 0.0,
                'fuzzy_aggressive': 0.0,
                'tfidf_similarity': 0.0,
                'semantic_openai': 0.0,
                'conservative_score': 0.0,
                'aggressive_score': 0.0
            }
        
        # Calculate different similarity metrics
        exact_score = self.exact_substring_match(cleaned_benchmark, system_content)
        fuzzy_score = self.fuzzy_match_aggressive(cleaned_benchmark, system_content)
        tfidf_score = self.tfidf_similarity(cleaned_benchmark, system_content)
        
        # OpenAI semantic similarity (optional for performance)
        semantic_score = 0.0
        if use_openai:
            semantic_score = self.semantic_similarity_openai(cleaned_benchmark, system_content)
        
        # Conservative score: average of exact and TF-IDF (more strict)
        conservative_score = (exact_score + tfidf_score) / 2
        
        # Aggressive score: weighted average favoring higher scores
        aggressive_score = (
            exact_score * 0.35 + 
           fuzzy_score * 0.15 + 
           #tfidf_score * 0.15 + 
            semantic_score * 0.5
        )
        
        return {
            'exact_match': exact_score,
            'fuzzy_aggressive': fuzzy_score,
            'tfidf_similarity': tfidf_score,
            'semantic_openai': semantic_score,
            'conservative_score': conservative_score,
            'aggressive_score': aggressive_score
        }
    
    def process_benchmark_cells(self, benchmark_df: pd.DataFrame, system_df: pd.DataFrame,
                              use_openai: bool = True, max_workers: int = 5) -> pd.DataFrame:
        """Process all benchmark cells and calculate similarity scores"""
        
        # Prepare system content once
        system_content = self.get_all_system_content(system_df)
        logger.info(f"System content length: {len(system_content)} characters")
        
        results = []
        
        # Process each cell in benchmark
        total_cells = benchmark_df.shape[0] * benchmark_df.shape[1]
        
        with tqdm(total=total_cells, desc="Processing cells") as pbar:
            for row_idx, row in benchmark_df.iterrows():
                for col_name in benchmark_df.columns:
                    cell_value = row[col_name]
                    
                    if pd.notna(cell_value) and str(cell_value).strip():
                        scores = self.analyze_cell_content(
                            cell_value, system_df, system_content, use_openai
                        )
                        
                        result = {
                            'row': row_idx,
                            'column': col_name,
                            'benchmark_content': str(cell_value)[:100] + "..." if len(str(cell_value)) > 100 else str(cell_value),
                            'content_length': len(str(cell_value)),
                            **scores
                        }
                        results.append(result)
                    
                    pbar.update(1)
        
        return pd.DataFrame(results)
    
    def generate_summary_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics and report"""
        
        if results_df.empty:
            return {"error": "No results to analyze"}
        
        summary = {
            'total_cells_analyzed': len(results_df),
            'average_scores': {
                'conservative_average': results_df['conservative_score'].mean(),
                'aggressive_average': results_df['aggressive_score'].mean(),
                'exact_match_average': results_df['exact_match'].mean(),
                'fuzzy_match_average': results_df['fuzzy_aggressive'].mean(),
                'tfidf_average': results_df['tfidf_similarity'].mean(),
                'semantic_average': results_df['semantic_openai'].mean()
            },
            'score_distribution': {
                'conservative': {
                    'high_match_90_plus': len(results_df[results_df['conservative_score'] >= 90]),
                    'good_match_70_89': len(results_df[(results_df['conservative_score'] >= 70) & (results_df['conservative_score'] < 90)]),
                    'moderate_match_50_69': len(results_df[(results_df['conservative_score'] >= 50) & (results_df['conservative_score'] < 70)]),
                    'low_match_below_50': len(results_df[results_df['conservative_score'] < 50])
                },
                'aggressive': {
                    'high_match_90_plus': len(results_df[results_df['aggressive_score'] >= 90]),
                    'good_match_70_89': len(results_df[(results_df['aggressive_score'] >= 70) & (results_df['aggressive_score'] < 90)]),
                    'moderate_match_50_69': len(results_df[(results_df['aggressive_score'] >= 50) & (results_df['aggressive_score'] < 70)]),
                    'low_match_below_50': len(results_df[results_df['aggressive_score'] < 50])
                }
            },
            'best_matches': results_df.nlargest(5, 'conservative_score')[['row', 'column', 'benchmark_content', 'conservative_score', 'aggressive_score']].to_dict('records'),
            'worst_matches': results_df.nsmallest(5, 'conservative_score')[['row', 'column', 'benchmark_content', 'conservative_score', 'aggressive_score']].to_dict('records')
        }
        
        return summary

def main():
    """Main execution function"""
    
    # Configuration
    OPENAI_API_KEY = "
    BENCHMARK_FILE = "Compare_outputs/benchmark.xlsx"  # Replace with your benchmark file path
    SYSTEM_FILE = "Compare_outputs/system.xlsx"      # Replace with your system file path
    USE_OPENAI = True  # Set to False to skip OpenAI semantic analysis for faster processing
    
    # Initialize comparator
    comparator = ExcelContentComparator(OPENAI_API_KEY)
    
    try:
        # Load Excel files
        print("Loading Excel files...")
        benchmark_df, system_df = comparator.load_excel_files(BENCHMARK_FILE, SYSTEM_FILE)
        
        # Process benchmark cells
        print("Processing benchmark cells...")
        results_df = comparator.process_benchmark_cells(
            benchmark_df, system_df, use_openai=USE_OPENAI
        )
        
        # Generate summary report
        print("Generating summary report...")
        summary = comparator.generate_summary_report(results_df)
        
        # Save detailed results
        results_df.to_excel("detailed_comparison_results.xlsx", index=False)
        print("Detailed results saved to 'detailed_comparison_results.xlsx'")
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK vs SYSTEM COMPARISON SUMMARY")
        print("="*60)
        print(f"Total cells analyzed: {summary['total_cells_analyzed']}")
        print(f"\nAverage Scores:")
        print(f"  Conservative Average: {summary['average_scores']['conservative_average']:.2f}%")
        print(f"  Aggressive Average: {summary['average_scores']['aggressive_average']:.2f}%")
        print(f"\nScore Breakdown (Conservative):")
        print(f"  High match (90%+): {summary['score_distribution']['conservative']['high_match_90_plus']} cells")
        print(f"  Good match (70-89%): {summary['score_distribution']['conservative']['good_match_70_89']} cells")
        print(f"  Moderate match (50-69%): {summary['score_distribution']['conservative']['moderate_match_50_69']} cells")
        print(f"  Low match (<50%): {summary['score_distribution']['conservative']['low_match_below_50']} cells")
        
        print(f"\nScore Breakdown (Aggressive):")
        print(f"  High match (90%+): {summary['score_distribution']['aggressive']['high_match_90_plus']} cells")
        print(f"  Good match (70-89%): {summary['score_distribution']['aggressive']['good_match_70_89']} cells")
        print(f"  Moderate match (50-69%): {summary['score_distribution']['aggressive']['moderate_match_50_69']} cells")
        print(f"  Low match (<50%): {summary['score_distribution']['aggressive']['low_match_below_50']} cells")
        
        print("\nTop 3 Best Matches:")
        for i, match in enumerate(summary['best_matches'][:3], 1):
            print(f"  {i}. Row {match['row']}, Col '{match['column']}': {match['conservative_score']:.1f}% (Conservative)")
        
        print("\nTop 3 Worst Matches:")
        for i, match in enumerate(summary['worst_matches'][:3], 1):
            print(f"  {i}. Row {match['row']}, Col '{match['column']}': {match['conservative_score']:.1f}% (Conservative)")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise

if __name__ == "__main__":
    main()