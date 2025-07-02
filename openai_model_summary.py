import requests
import json
import re, sys
import csv, os
import pandas as pd
from dotenv import load_dotenv
from retrieval import improved_hybrid_search
load_dotenv()

# Get API Key securely
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY = 

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
        ]
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

def create_summary_from_extracted_sentences(extracted_sentences):
    """
    Creates a short summary statement from extracted sentences while preserving key verbatim content
    """
    summary_prompt = f"""
    Create a short, concise summary statement of less than 200 words from the following extracted sentences. 

    INSTRUCTIONS:
    1. Preserve verbatim content for:
       - Key Performance Indicators (KPIs) and their names
       - Specific metrics, numbers, and percentages
       - Company names, product names, and proper nouns
       - Technical terms and industry-specific terminology

    2. "Try to make the summary combine the information logically."
    3. "Do not use prior knowledge or make assumptions not grounded in the Extracted Sentences."
    4. "Do not include hallucinations, personal opinions, but make sure the answer still follows logic and is correct business-wise."
    5. "Each answer should directly address the specific question asked."
    6. "Provide the answer in a structured way, as appropriate for the question."
    7. "Make sure the summary is concise and to the point, while still capturing the essence of the information."
    
    Extracted Sentences:
    {extracted_sentences}
    
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
                
                # Create summary while preserving verbatim content
                summary = create_summary_from_extracted_sentences(combined_sentences)
                processed_summaries[query] = {
                    'original_sentences': all_sentences,
                    'combined_sentences': combined_sentences,
                    'summary': summary
                }
                
                print(f"Generated summary: {summary}")
                print("-" * 80)
            else:
                processed_summaries[query] = {
                    'original_sentences': [],
                    'combined_sentences': "",
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
    file_exists = os.path.isfile("outputs/openairesponses_from_excelv2.3.csv")
    with open("outputs/openairesponses_from_excelv2.2.csv", mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Query", "Original_Sentences_Count", "Summary", "Score"])

        for query, data in processed_summaries.items():
            print(f"\n============================")
            print(f"Processing Query: {query}")
            print("============================")

            summary_context = data['summary']
            
            if summary_context and summary_context != "No extracted sentences found for this query":
                # Generate the prompt using the summary
                #openai_prompt = generate_openai_prompt_with_summary(query, summary_context)

                print(f"\n[INFO] Generating response using OpenAI model for query: {query}")
                #model_response = generate_text(prompt=openai_prompt)

                print("Query | Summary")
                print("------------------------------------------------------------")
                print(f'"{query}" | {summary_context}')

                writer.writerow([
                    query, 
                    len(data['original_sentences']), 
                    summary_context, 
                    "Processed"
                ])
                
            else:
                print(f'"{query}" | "No relevant content found" | Unable to score')
                writer.writerow([query, 0, "No extracted sentences found", "No relevant content found", "Unable to score"])

    print("\n[INFO] All responses saved to openairesponses_from_excel.csv")


if __name__ == "__main__":
    # Check if Excel processing is requested
    if len(sys.argv) == 2 and sys.argv[1].endswith('.csv'):
        # Process Excel file mode
        excel_file_path = sys.argv[1]
        print(f"Processing Excel file: {excel_file_path}")
        process_excel_results(excel_file_path)

