import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Load data
manual_df = pd.read_csv(r"C:\Users\mitra\Downloads\Manual_Summary.csv")
system_df = pd.read_csv(r"C:\Users\mitra\Downloads\pdf_analysis_rag_poc\outputs\summary_RILq1_comp.csv")

manual_df = manual_df[["Query", "Summary"]].dropna().reset_index(drop=True)
system_df = system_df[["Query", "Summary"]].dropna().reset_index(drop=True)

# Load model
model = SentenceTransformer("all-mpnet-base-v2")

# Similarity metrics
def calculate_soft_cosine_similarity(text1, text2):
    embeddings = model.encode([text1, text2], convert_to_tensor=True, normalize_embeddings=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    #score = 100 / (1 + np.exp(-8 * (similarity - 0.3)))
    score = similarity * 100

    return round(score, 2)

def calculate_rouge_l_similarity(text1, text2):
    def lcs_length(s1, s2):
        words1, words2 = s1.lower().split(), s2.lower().split()
        m, n = len(words1), len(words2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    lcs = lcs_length(text1, text2)
    words1, words2 = text1.lower().split(), text2.lower().split()
    recall = lcs / len(words1) if words1 else 0
    precision = lcs / len(words2) if words2 else 0
    if recall + precision == 0:
        return 0.0
    f1 = 2 * (recall * precision) / (recall + precision)
    return round(f1 * 100, 2)

def calculate_bleu_similarity(text1, text2):
    from collections import Counter
    def get_ngrams(text, n):
        words = text.lower().split()
        return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

    scores = []
    for n in [1, 2]:
        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)
        counter1, counter2 = Counter(ngrams1), Counter(ngrams2)
        matches = sum((counter1 & counter2).values())
        total = len(ngrams1)
        if total > 0:
            scores.append(matches / total)
    if not scores:
        return 0.0
    bleu = np.exp(np.mean(np.log(np.array(scores) + 1e-10)))
    return round(bleu * 100, 2)

def calculate_ensemble_similarity(text1, text2):
    c = calculate_soft_cosine_similarity(text1, text2)
    r = calculate_rouge_l_similarity(text1, text2)
    b = calculate_bleu_similarity(text1, text2)
    score = c
    return round(score, 2)

def interpret_score(score):
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Very Good"
    elif score >= 50:
        return "Good"
    elif score >= 30:
        return "Fair"
    else:
        return "Poor"

# Embed questions
manual_qs = manual_df["Query"].tolist()
system_qs = system_df["Query"].tolist()

manual_embs = model.encode(manual_qs, convert_to_tensor=True)
system_embs = model.encode(system_qs, convert_to_tensor=True)

results = []

# Compare
for i, m_emb in enumerate(manual_embs):
    sims = util.cos_sim(m_emb, system_embs)[0]
    best_idx = torch.argmax(sims).item()
    best_score = sims[best_idx].item()

    m_query = manual_qs[i]
    s_query = system_qs[best_idx]
    m_summary = manual_df.loc[i, "Summary"]
    s_summary = system_df.loc[best_idx, "Summary"]

    sim_score = calculate_ensemble_similarity(m_summary, s_summary)
    interpretation = interpret_score(sim_score)

    results.append({
        "Manual Query": m_query,
        "System Query": s_query,
        "Query Similarity": round(best_score, 2),
        "Summary Similarity": sim_score,
        "Interpretation": interpretation
    })

    # Print results
    print(f"\n🔎 Matched Question {i+1}")
    print(f"🟠 Manual Query: {m_query}")
    print(f"🔵 System Query: {s_query}")
    print(f"📘 Query Similarity Score: {round(best_score * 100, 2)}%")
    print(f"📄 Summary Similarity Score: {sim_score}% - {interpretation}")
    print("-" * 80)

# Save results
result_df = pd.DataFrame(results)
result_df.to_csv("summary_similarity_comparison.csv", index=False)
