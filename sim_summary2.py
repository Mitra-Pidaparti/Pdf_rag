import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load data
manual_df = pd.read_csv(r"C:\Users\mitra\Downloads\Manual_Summary.csv")
system_df = pd.read_csv(r"C:\Users\mitra\Downloads\pdf_analysis_rag_poc\outputs\summary_RILq1_comp.csv")

manual_df = manual_df[["Query", "Summary"]].dropna().reset_index(drop=True)
system_df = system_df[["Query", "Summary"]].dropna().reset_index(drop=True)

# Load stronger semantic model
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Function to calculate cosine similarity between summaries
def calculate_summary_cosine_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)

# Embed queries
manual_qs = manual_df["Query"].tolist()
system_qs = system_df["Query"].tolist()

manual_embs = model.encode(manual_qs, convert_to_tensor=True)
system_embs = model.encode(system_qs, convert_to_tensor=True)

results = []

# Match queries and compare summaries
for i, m_emb in enumerate(manual_embs):
    sims = util.cos_sim(m_emb, system_embs)[0]
    best_idx = torch.argmax(sims).item()
    best_score = sims[best_idx].item()

    m_query = manual_qs[i]
    s_query = system_qs[best_idx]
    m_summary = manual_df.loc[i, "Summary"]
    s_summary = system_df.loc[best_idx, "Summary"]


    print(s_summary)
    print("Manual_SUmmary")
    print(m_summary)
    summary_score = calculate_summary_cosine_similarity(m_summary, s_summary)

    results.append({
        "Manual Query": m_query,
        "System Query": s_query,
        "Query Similarity": round(best_score * 100, 2),
        "Summary Similarity": summary_score
    })

    # Print results
    print(f"\n🔎 Matched Question {i+1}")
    print(f"🟠 Manual Query: {m_query}")
    print(f"🔵 System Query: {s_query}")
    print(f"📘 Query Similarity Score: {round(best_score * 100, 2)}%")
    print(f"📄 Summary Cosine Similarity Score: {summary_score}%")
    print("-" * 80)

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv("summary_cosine_similarity.csv", index=False)
