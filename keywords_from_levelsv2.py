import pandas as pd
import openai
import json
import os

from dotenv import load_dotenv


load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
INPUT_FILE = "levels_novartis.xlsx"
OUTPUT_FILE = "novartis_keywords_by_question.xlsx"

class KeywordExtractorByQuestion:
    def __init__(self, api_key: str):
        if not api_key or "YOUR_OPENAI_API_KEY" in api_key:
            raise ValueError("Set OPENAI_API_KEY in your environment or script.")
        openai.api_key = api_key

    def _parse(self, file_path: str) -> pd.DataFrame:
        df = pd.read_excel(file_path)
        df = df.dropna(subset=['Question'])
        df_long = df.melt(
            id_vars=['Question'],
            value_vars=[col for col in df.columns if col.startswith('Response')],
            var_name='Level',
            value_name='Level_Description'
        )
        df_long = df_long.dropna(subset=['Level_Description'])
        df_long['Level'] = df_long['Level'].str.extract('(\d+)').astype(int)
        df_long = df_long.sort_values(['Question', 'Level'])
        return df_long

    def _extract_keywords_llm(self, level_description: str):
        prompt = f"""
You are an expert in corporate strategy, business communication, and language processing.

Your task is to extract **all relevant keywords and keyphrases** from maturity level text provided below. This should include not only exact terms but also related terms, synonyms, and commonly used variations in professional or corporate contexts.

From this text:
"{level_description}"

Instructions:

1. **Extract all key concepts, topics, and terms** that capture the essence of the input text.
2. For each keyword or keyphrase, include:
   - Synonyms
   - Near-synonyms
   - Common variations or alternate phrasings
   - Singular/plural forms (only when meaning changes

4. Avoid generic words or stopwords (e.g., "the", "and", "is").
5. Return the result as a **flat JSON array of strings**, with **no nesting or duplicates**.
"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
                max_tokens=200
            )
            content = json.loads(response.choices[0].message.content)
            for val in content.values():
                if isinstance(val, list):
                    return val
            return [str(content)]
        except Exception as e:
            print(f"Failed LLM keyword extraction: {e}")
            return ["ERROR"]

    def run(self):
        print(f"Reading: {INPUT_FILE}")
        df_long = self._parse(INPUT_FILE)
        print(f"Parsed {len(df_long)} (question, level) entries.")

        # Extract keywords per (question, level)
        keywords_pairs = []
        for idx, row in df_long.iterrows():
            print(f"Q: {row['Question'][:40]}... Level {row['Level']}")
            level_text = row['Level_Description']
            extracted_keywords = self._extract_keywords_llm(level_text)
            keywords_pairs.append((row['Question'], extracted_keywords))

        # Aggregate: collect all keywords across all levels, deduplicated, per question
        question_to_keywords = {}
        for question, keylist in keywords_pairs:
            if question not in question_to_keywords:
                question_to_keywords[question] = set()
            question_to_keywords[question].update([k.strip() for k in keylist if isinstance(k, str)])
        
        rows = []
        for question, keywords in question_to_keywords.items():
            rows.append({'Question': question, 'Keywords': sorted(list(keywords))})

        df_questions = pd.DataFrame(rows)
        df_questions.to_excel(OUTPUT_FILE, index=False, engine='openpyxl')
        print(f"Saved to {OUTPUT_FILE}")
        return df_questions

if __name__ == "__main__":
    KeywordExtractorByQuestion(api_key=OPENAI_API_KEY).run()
