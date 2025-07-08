import requests
import json
import re, sys
import csv, os
import pandas as pd
from dotenv import load_dotenv
#from retrieval import improved_hybrid_search
load_dotenv()

# Get API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY is missing in the .env file.")

def generate_text(prompt: str, model: str = "gpt-4.1-2025-04-14"):
    """
    Sends a request to OpenAI's API using GPT-4o and returns the generated response.
    """
    url = "https://api.openai.com/v1/chat/completions"  # OpenAI API endpoint
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an AI assistant. Provide detailed responses while preserving key terms."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1 # <-- Add or modify this line to control creativity
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_json = response.json()
        response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
        
        # Remove content between <think> and </think> tags to remove thinking output
        final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
        
        return final_answer
    else:
        return f"Error: Request failed with status code {response.status_code}, {response.text}"

def extract_most_representative_statement(summary, query):
    """
    Extracts the most representative statement from a generated summary based on the query.
    """
    if not summary:
        return "No summary available"
    
    selection_prompt = f"""
You are tasked with selecting the MOST representative statement from the following summary that best answers the query: "{query}"

SELECTION CRITERIA:
1. Choose the sentence that most directly and comprehensively addresses the query.
2. Select sentence that provide the most strategic or high-level insight.
3. Avoid generic or vague sentences.
4. If multiple sentences are equally relevant, choose the one with the most specific information.

SUMMARY:
{summary}

INSTRUCTIONS:
- Return ONLY the selected sentence exactly as written.
- Do not add any commentary, explanations, or modifications.
- If no sentence adequately addresses the query, return the most relevant one available.

Query: {query}

Most representative statement:
"""
    return generate_text(selection_prompt)

def create_summary_from_extracted_sentences(extracted_sentences, query):
    """
    Creates a hierarchical, top-down summary from extracted sentences with improved structure
    """
    summary_prompt = f"""
You are creating an executive summary for an annual report analysis. Your task is to synthesize the extracted content into a coherent, top-down narrative that directly answers: "{query}"

"STRUCTURE REQUIREMENTS (use this internally to guide the flow — do not include section headings in the output):"

1. STRATEGIC POSITIONING (1-2 sentences): Open with the company's overarching approach, philosophy, or strategic stance in relation to the query topic. Use phrases like "has a structured approach to...", "actively investing in...", "embodies a vision..."

2. STRATEGIC RATIONALE (1-2 sentences): Explain the underlying reasoning or broader context that drives this approach. Connect to business logic, market dynamics, or stakeholder value.

3. SPECIFIC MECHANISMS & IMPLEMENTATIONS (4-5 sentences): Detail the concrete ways the company executes this strategy, including programs, technologies, metrics, partnerships, or initiatives. Preserve exact figures, percentages, and proper nouns verbatim. Combine similar points to eliminate redundancy and ensure a flowing narrative.

4. In case different SBU(Strategic Business Units) are mentioned, ensure to include them in the summary, with each SBU's details separated clearly.

5. INTEGRATION & IMPACT (1-2 sentences): Conclude by showing how all elements connect and the resulting outcomes or implications for the business.

TONE & GUIDELINES:
- Maintain an analytical tone with subtle narrative insight, suitable for C-suite executives
- Ensure the summary is concise and does not exceed 150-200 words
- Make sure not to repeat "structured approach" or similar phrases multiple times
- ONLY use information present in the extracted content. Do NOT infer or introduce new claims
- Synthesize thematically and eliminate duplication
- For the question:"What are the key Metrics / KPIs / key performance indicators being tracked related to the innovation portfolio and overall business performance?" Focus on the innovation portfolio mainly.
- Dont repeat the word SBU too much, the info should be seamlessly integrated
Query: {query}

Extracted Content:
{extracted_sentences}

Generate a well-structured summary following the exact pattern above:
"""
    
    return generate_text(summary_prompt)

def load_and_process_excel(excel_file_path):
    """
    Load the Excel file and process extracted sentences for each unique query
    """
    try:
        # Read the Excel file
        df = pd.read_csv(excel_file_path)
        
        # Group by the first column (assuming it contains queries)
        query_column = df.columns[0]  # First column
        extracted_sentences_column = 'Extracted Sentences'  # Specified column name
        
        if extracted_sentences_column not in df.columns:
            raise ValueError(f"Column '{extracted_sentences_column}' not found in the Excel file")
        
        # Group by unique queries
        grouped = df.groupby(query_column)
        
        processed_summaries = {}
        
        for query, group in grouped:
            print(f"\nProcessing query: {query}")
            
            # Combine all extracted sentences for this query
            all_sentences = []
            for sentence in group[extracted_sentences_column].dropna():
                if sentence.strip():  # Only add non-empty sentences
                    all_sentences.append(sentence.strip())
            
            if all_sentences:
                # Join sentences with proper separation
                combined_sentences = "\n".join([f"- {sentence}" for sentence in all_sentences])
                
                print(f"Combined sentences for '{query}':")
                print(combined_sentences)
                
                # Extract the most representative statement
                representative_statement = extract_most_representative_statement(all_sentences, query)
                
                # Create summary while preserving verbatim content
                summary = create_summary_from_extracted_sentences(combined_sentences, query)
                processed_summaries[query] = {
                    'original_sentences': all_sentences,
                    'combined_sentences': combined_sentences,
                    'representative_statement': representative_statement,
                    'summary': summary
                }
                
                print(f"Most representative statement: {representative_statement}")
                print(f"Generated summary: {summary}")
                print("-" * 80)
            else:
                processed_summaries[query] = {
                    'original_sentences': [],
                    'combined_sentences': "",
                    'representative_statement': "No sentences available",
                    'summary': "No extracted sentences found for this query"
                }
        
        return processed_summaries
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return {}


def process_excel_results(excel_file_path):
    """
    Process results from Excel file and generate responses
    """
    # Load and process the Excel file
    processed_summaries = load_and_process_excel(excel_file_path)
    
    if not processed_summaries:
        print("No data processed from Excel file.")
        return
    
    # Create output file
    file_exists = os.path.isfile("outputs/summary_RIL2.csv")
    with open("outputs/summary_RIL.csv", mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Query", "Original_Sentences_Count", "Most_Representative_Statement", "Summary", "Score"])

        for query, data in processed_summaries.items():
            print(f"\n============================")
            print(f"Processing Query: {query}")
            print("============================")

            summary_context = data['summary']
            representative_statement = data['representative_statement']
            
            if summary_context and summary_context != "No extracted sentences found for this query":
                # Generate the prompt using the summary
                #openai_prompt = generate_openai_prompt_with_summary(query, summary_context)

                print(f"\n[INFO] Generating response using OpenAI model for query: {query}")
                #model_response = generate_text(prompt=openai_prompt)

                print("Query | Most Representative Statement | Summary")
                print("------------------------------------------------------------")
                print(f'"{query}" | {representative_statement} | {summary_context}')

                writer.writerow([
                    query, 
                    len(data['original_sentences']), 
                    representative_statement,
                    summary_context, 
                    "Processed"
                ])
                
            else:
                print(f'"{query}" | "No representative statement found" | "No relevant content found" | Unable to score')
                writer.writerow([query, 0, "No sentences available", "No extracted sentences found", "Unable to score"])

    print("\n[INFO] All responses saved to summary_RIL.csv")


if __name__ == "__main__":
    # Check if Excel processing is requested
    if len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        # Process Excel file mode
        excel_file_path = sys.argv[1]
        print(f"Processing Excel file: {excel_file_path}")
        process_excel_results(excel_file_path)