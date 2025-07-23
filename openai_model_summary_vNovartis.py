import requests
import json
import re, sys
import csv, os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Get API Key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("Error: OPENAI_API_KEY is missing in the .env file.")

def generate_text(prompt: str, model: str = "o3"): #model name
    """
    Sends a request to OpenAI's API using o3 and returns the generated response.
    """
    url = "https://api.openai.com/v1/chat/completions"
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
        #"temperature": 0.2
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_json = response.json()
        response_content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response generated.")
        
        # Remove content between <think> and </think> tags
        final_answer = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
        
        return final_answer
    else:
        return f"Error: Request failed with status code {response.status_code}, {response.text}"


def create_summary_from_extracted_sentences(extracted_sentences, query):
    """
    Creates a hierarchical, top-down summary from extracted sentences with improved structure
    """
    summary_prompt = f"""
TASK DESCRIPTION

Synthesize the extracted content into a concise, coherent, and executive-level narrative that directly addresses the query:
"{query}"

    Each input chunk is separated by two newlines. They are in a random order and not sequential. Do not ignore or skip any relevant chunk.

    Ensure that you:
    Preserve verbatim any critical elements including:
    - Named initiatives or programs
    - Strategic terminology (e.g., purpose statements, frameworks)
    - Specific metrics or KPIs (counts, percentages, figures)
    - Names of business units, geographies, or partners
    - Do not paraphrase these elements. Use exact wording as given.

Strictly avoid hallucination. Do not infer, invent, or introduce content not explicitly present in the input. If information appears incomplete or ambiguous, flag it for review rather than guessing.
If highly relevant information is not included verbatim, ensure its core idea is faithfully and clearly represented.


STRUCTURE -use this internally to guide the flow — NEVER include section headings in the output:

1. STRATEGIC POSITIONING:
Begin with the company's overarching philosophy or strategic stance on the topic.

2. STRATEGIC RATIONALE:
Explain the key drivers, motivations, or contextual factors behind the chosen approach.

3. MECHANISMS & EXECUTION & INITIATIVES:
Describe the company's initiatives, technologies, partnerships, and measurable outcomes related to the query, in a logical way.
Include exact data (figures, percentages, names) verbatim. Merge related points to ensure clarity and flow.

4. STRATEGIC BUSINESS UNITS (SBUs):(only if applicable and relevant to the query, else ignore):
- Dont include the heading, just present the information
If multiple Strategic Business Units (SBUs) are referenced, present each SBU's input clearly, with separate summaries or segments as needed.

5. INTEGRATION & IMPACT (1-2 sentences):
Conclude by synthesizing the overall impact—how the approach, rationale, and mechanisms align to drive business outcomes.

TONE & GUIDELINES

    -Each line should be a logical continuation of the previous line. Maintain clarity and readability
    -Maintain a precise, analytical tone with narrative flow, targeting C-suite decision-makers.
    -Ensure grammar, spelling, and punctuation are flawless.
    -Make sure to include means(tools + activities), initiatives, Tactical execution
    -Keep the summary between 200-250 words. Prioritize clarity and brevity.
    -Try to  include all important information explicitly present in the extracted content.
    -Do not infer, speculate, or add assumptions. Aim to preserve the breadth of relevant points.
    -Group related insights thematically. Remove duplication to ensure clarity and flow.

Query: {query}

Extracted Content:
{extracted_sentences}

Generate a well-structured summary following the exact pattern above:
"""
    
    return generate_text(summary_prompt)

def load_and_process_file(file_path):
    """
    Load the file (Excel or CSV) and process extracted sentences for each column (question)
    """
    try:
        # Determine file type and read accordingly
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Loaded file with shape: {df.shape}")
        print(f"Columns found: {list(df.columns)}")
        
        processed_summaries = {}
        
        # Process each column as a separate query
        for column in df.columns:
            query = column.strip()
            print(f"\nProcessing column/query: {query}")
            
            # Get all non-null, non-empty values from this column
            column_values = df[column].dropna()
            
            # Filter out empty strings and whitespace-only strings
            valid_sentences = []
            for value in column_values:
                if isinstance(value, str) and value.strip():
                    valid_sentences.append(value.strip())
            
            if valid_sentences:
                # Join sentences with proper separation
                combined_sentences = "\n\n".join([f"- {sentence}" for sentence in valid_sentences])
                
                print(f"Found {len(valid_sentences)} valid sentences for '{query}'")
                print("Sample sentences:")
                for i, sentence in enumerate(valid_sentences[:3]):  # Show first 3 as sample
                    print(f"  {i+1}. {sentence[:100]}...")
                
                # Create summary while preserving verbatim content
                summary = create_summary_from_extracted_sentences(combined_sentences, query)
                processed_summaries[query] = {
                    'original_sentences': valid_sentences,
                    'combined_sentences': combined_sentences,
                    'summary': summary
                }
                
                print(f"Generated summary: {summary}")
                print("-" * 80)
            else:
                print(f"No valid sentences found for column '{query}'")
                processed_summaries[query] = {
                    'original_sentences': [],
                    'combined_sentences': "",
                    'summary': "No extracted sentences found for this query"
                }
        
        return processed_summaries
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return {}

def process_file_results(file_path):
    """
    Process results from file and generate responses
    """
    # Load and process the file
    processed_summaries = load_and_process_file(file_path)
    
    if not processed_summaries:
        print("No data processed from file.")
        return
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Create output file
    output_file = "outputs/summary_novartisQ2_nr.csv"
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, mode="a", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Query", "Original_Sentences_Count", "Summary", "Score"])

        for query, data in processed_summaries.items():
            print(f"\n============================")
            print(f"Processing Query: {query}")
            print("============================")

            summary_context = data['summary']
            
            if summary_context and summary_context != "No extracted sentences found for this query":
                print(f"\n[INFO] Generating response using OpenAI model for query: {query}")

                print("Query| Summary")
                print("------------------------------------------------------------")
                print(f'"{query}"  | {summary_context}')

                writer.writerow([
                    query, 
                    len(data['original_sentences']), 
                    summary_context, 
                    "Processed"
                ])
                
            else:
                print(f'"{query}" | "No relevant content found" | Unable to score')
                writer.writerow([query, 0, "No sentences available", "Unable to score"])

    print(f"\n[INFO] All responses saved to {output_file}")

if __name__ == "__main__":
    # Check if file processing is requested
    if len(sys.argv) == 2 and (sys.argv[1].endswith('.csv') or sys.argv[1].endswith('.xlsx') or sys.argv[1].endswith('.xls')):
        # Process file mode
        file_path = sys.argv[1]
        print(f"Processing file: {file_path}")
        process_file_results(file_path)
    else:
        print("Usage: python script.py <file_path>")
        print("Supported formats: .csv, .xlsx, .xls")