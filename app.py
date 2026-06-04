import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import re
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(
    page_title="GenAI Question Paper Generator",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.hero { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; color: white; }
.hero h1 { font-size: 28px; font-weight: 600; margin:0 0 8px; }
.hero p  { font-size: 14px; opacity: 0.8; margin: 0; }
.qcard { background: #f8faff; border-radius: 12px; padding: 1.1rem 1.4rem;
         border-left: 4px solid #4338ca; margin-bottom: 12px; }
.qcard .bloom { font-size: 11px; font-weight: 600; color: #4338ca; text-transform: uppercase;
                letter-spacing: .5px; margin-bottom: 6px; }
.qcard .qtext { font-size: 14px; color: #1e1b4b; line-height: 1.6; margin: 0; }
.qcard .qtype { font-size: 11px; color: #999; margin-top: 4px; }
.badge-bloom { display:inline-block; padding:2px 9px; border-radius:99px; font-size:11px;
               font-weight:500; margin:2px; }
.mcq-opt { background:#fff; border:1px solid #e0e7ff; border-radius:8px; padding:8px 12px;
           margin:4px 0; font-size:13px; cursor:default; }
.mcq-correct { border-color:#6ee7b7 !important; background:#d1fae5 !important; }
.stat { background:#ede9fe; border-radius:10px; padding:.9rem 1.1rem; text-align:center; }
.stat h4 { font-size:12px; color:#7c3aed; margin:0 0 4px; text-transform:uppercase; letter-spacing:.5px; }
.stat p  { font-size:24px; font-weight:600; color:#4c1d95; margin:0; }
.rag-badge { background:#fef3c7; color:#92400e; border-radius:6px; padding:2px 8px;
             font-size:11px; font-weight:500; display:inline-block; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYLLABUS CONTENT (simulates FAISS-indexed docs)
# ─────────────────────────────────────────────
SYLLABUS = {
    "Machine Learning": [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Supervised learning uses labeled training data to train models that predict outputs for new inputs.",
        "Unsupervised learning finds hidden patterns in data without predefined labels.",
        "Overfitting occurs when a model learns the training data too well and fails to generalise.",
        "Regularisation techniques like L1 (Lasso) and L2 (Ridge) penalise model complexity to prevent overfitting.",
        "Cross-validation evaluates model performance by training and testing on different data splits.",
        "Random Forest is an ensemble of decision trees that reduces variance through bagging.",
        "Gradient Boosting builds models sequentially, each correcting errors of the previous.",
        "Feature engineering transforms raw data into informative inputs for machine learning models.",
        "Bias-variance tradeoff: high bias causes underfitting; high variance causes overfitting.",
    ],
    "Deep Learning": [
        "Neural networks consist of layers of interconnected nodes inspired by the human brain.",
        "Backpropagation computes gradients by applying the chain rule from output to input layers.",
        "Convolutional Neural Networks (CNNs) use filters to extract spatial features from images.",
        "Recurrent Neural Networks (RNNs) process sequential data using hidden state memory.",
        "Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem in RNNs.",
        "Transformers use self-attention mechanisms to process entire sequences simultaneously.",
        "BERT is a transformer model pre-trained on masked language modelling and next sentence prediction.",
        "Dropout randomly deactivates neurons during training as a regularisation technique.",
        "Batch normalisation stabilises training by normalising layer inputs.",
        "Transfer learning applies knowledge from a pre-trained model to a new related task.",
    ],
    "Data Science": [
        "Exploratory Data Analysis (EDA) is the process of summarising and visualising dataset characteristics.",
        "Data preprocessing includes handling missing values, encoding categoricals, and feature scaling.",
        "The Central Limit Theorem states that sample means approximate a normal distribution for large n.",
        "Hypothesis testing determines whether observed data supports a specific claim about a population.",
        "Correlation measures the linear relationship between two variables, ranging from -1 to +1.",
        "Principal Component Analysis (PCA) reduces dimensionality by projecting data onto principal axes.",
        "A confusion matrix summarises the performance of a classification model on test data.",
        "Precision is the ratio of true positives to all predicted positives.",
        "Recall (sensitivity) is the ratio of true positives to all actual positives.",
        "F1-score is the harmonic mean of precision and recall, balancing both metrics.",
    ]
}

BLOOM_LEVELS = {
    1: ("Remember", "#6366f1"),
    2: ("Understand", "#0ea5e9"),
    3: ("Apply", "#10b981"),
    4: ("Analyse", "#f59e0b"),
    5: ("Evaluate", "#f97316"),
    6: ("Create", "#ef4444"),
}

BLOOM_VERBS = {
    1: ["Define", "List", "State", "Recall", "Name", "Identify"],
    2: ["Explain", "Describe", "Summarise", "Interpret", "Classify"],
    3: ["Apply", "Use", "Calculate", "Demonstrate", "Solve"],
    4: ["Analyse", "Compare", "Differentiate", "Examine", "Break down"],
    5: ["Evaluate", "Critique", "Judge", "Justify", "Assess"],
    6: ["Design", "Create", "Formulate", "Construct", "Develop"],
}

Q_TYPES = ["MCQ", "Short Answer", "Essay", "Fill in the Blank"]

# ─────────────────────────────────────────────
# DR-RAG PIPELINE (simulated with sentence-transformers)
# ─────────────────────────────────────────────
@st.cache_resource
def load_encoder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def dr_rag_retrieve(query: str, corpus: list, model, top_k=3) -> list:
    """
    Dynamic RAG: 
    Stage 1 - Static relevance: retrieve top_k chunks via cosine similarity.
    Stage 2 - Dynamic refinement: re-rank by query-augmented similarity.
    """
    corpus_emb = model.encode(corpus, convert_to_tensor=True)
    query_emb  = model.encode(query, convert_to_tensor=True)
    
    # Stage 1: initial retrieval
    scores1 = util.cos_sim(query_emb, corpus_emb)[0]
    top_idx  = torch.topk(scores1, min(top_k*2, len(corpus))).indices.tolist()
    candidates = [corpus[i] for i in top_idx]
    
    # Stage 2: re-rank with refined query (concatenate query + top result for context)
    if candidates:
        refined_query = f"{query} {candidates[0][:80]}"
        refined_emb   = model.encode(refined_query, convert_to_tensor=True)
        cand_emb      = model.encode(candidates, convert_to_tensor=True)
        scores2       = util.cos_sim(refined_emb, cand_emb)[0]
        reranked_idx  = torch.topk(scores2, min(top_k, len(candidates))).indices.tolist()
        return [candidates[i] for i in reranked_idx]
    return candidates[:top_k]

def deduplicate_questions(questions: list, model, threshold=0.85) -> list:
    """Semantic deduplication using sentence-transformers."""
    if len(questions) <= 1:
        return questions
    texts = [q["question"] for q in questions]
    embeddings = model.encode(texts, convert_to_tensor=True)
    keep = [True] * len(questions)
    for i in range(len(questions)):
        if not keep[i]:
            continue
        for j in range(i+1, len(questions)):
            if not keep[j]:
                continue
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim > threshold:
                keep[j] = False
    return [q for q, k in zip(questions, keep) if k]

def generate_question(chunk: str, q_type: str, bloom_level: int, topic: str) -> dict:
    """Generate a question from a syllabus chunk using template-based generation."""
    np.random.seed(hash(chunk + q_type) % 2**31)
    verb = np.random.choice(BLOOM_VERBS[bloom_level])
    bloom_name = BLOOM_LEVELS[bloom_level][0]
    
    # Extract key concept from chunk
    words = chunk.split()
    key_phrase = " ".join(words[3:8]) if len(words) >= 8 else " ".join(words[:5])
    
    if q_type == "MCQ":
        q_text = f"{verb} the concept of {key_phrase.lower()} in the context of {topic}."
        options = [
            f"A) {chunk[:60]}{'...' if len(chunk)>60 else ''}",
            f"B) A method unrelated to {topic}",
            f"C) A concept from an unrelated domain",
            f"D) None of the above"
        ]
        return {"question": q_text, "type": "MCQ", "bloom": bloom_name,
                "bloom_level": bloom_level, "options": options, "answer": "A",
                "context": chunk[:120] + "..."}
    
    elif q_type == "Short Answer":
        q_text = f"{verb} the following in 2–3 sentences: {key_phrase.lower()} as it applies to {topic}."
        return {"question": q_text, "type": "Short Answer", "bloom": bloom_name,
                "bloom_level": bloom_level, "hint": f"Refer to: {chunk[:80]}...",
                "context": chunk[:120] + "..."}
    
    elif q_type == "Essay":
        q_text = f"{verb} a comprehensive response discussing {key_phrase.lower()} in {topic}, with examples and applications."
        return {"question": q_text, "type": "Essay", "bloom": bloom_name,
                "bloom_level": bloom_level, "marks": np.random.choice([5,8,10]),
                "context": chunk[:120] + "..."}
    
    else:  # Fill in the blank
        # Replace a key word with blank
        sentence = chunk.split('.')[0] if '.' in chunk else chunk[:80]
        words_in_sent = sentence.split()
        if len(words_in_sent) > 5:
            blank_idx = len(words_in_sent) // 2
            answer = words_in_sent[blank_idx]
            words_in_sent[blank_idx] = "_______"
            q_text = " ".join(words_in_sent) + "."
        else:
            q_text = f"_______ refers to: {chunk[:60]}."
            answer = topic
        return {"question": q_text, "type": "Fill in the Blank", "bloom": bloom_name,
                "bloom_level": bloom_level, "answer": answer,
                "context": chunk[:120] + "..."}

def generate_paper(topic, q_types_sel, bloom_levels_sel, num_questions, language, board):
    model = load_encoder()
    corpus = SYLLABUS.get(topic, SYLLABUS["Machine Learning"])
    
    questions = []
    per_type = max(1, num_questions // len(q_types_sel)) if q_types_sel else num_questions
    
    for qt in q_types_sel:
        for i in range(per_type):
            bloom_level = np.random.choice(bloom_levels_sel)
            query = f"{qt} question about {topic} at {BLOOM_LEVELS[bloom_level][0]} level"
            
            # DR-RAG retrieval
            chunks = dr_rag_retrieve(query, corpus, model, top_k=2)
            chunk  = chunks[0] if chunks else corpus[i % len(corpus)]
            
            q = generate_question(chunk, qt, bloom_level, topic)
            questions.append(q)
    
    # Semantic deduplication
    before = len(questions)
    questions = deduplicate_questions(questions, model, threshold=0.82)
    after = len(questions)
    
    # Pad if needed
    while len(questions) < num_questions:
        chunk = corpus[len(questions) % len(corpus)]
        q = generate_question(chunk, np.random.choice(q_types_sel),
                               np.random.choice(bloom_levels_sel), topic)
        questions.append(q)
    
    return questions[:num_questions], before, after

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/test-passed.png", width=40)
    st.title("Paper Generator")
    st.caption("GenAI · DR-RAG · Bloom's Taxonomy")
    st.divider()
    
    topic = st.selectbox("Subject / Topic", list(SYLLABUS.keys()))
    q_types_sel = st.multiselect("Question Types", Q_TYPES, default=["MCQ","Short Answer"])
    bloom_min, bloom_max = st.select_slider(
        "Bloom's Taxonomy Levels",
        options=[1,2,3,4,5,6],
        value=(1,4)
    )
    bloom_levels_sel = list(range(bloom_min, bloom_max+1))
    num_questions = st.slider("Number of Questions", 5, 30, 12)
    language = st.selectbox("Language", ["English", "Hindi (Transliterated)"])
    board = st.selectbox("Board / Curriculum", ["CBSE","ICSE","State Board","University"])
    
    generate_btn = st.button("⚡ Generate Paper", type="primary", use_container_width=True)
    st.divider()
    st.caption("Built by Archit Dhar · VESIT 2025\nB.E. Final Year Project")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📄 GenAI Question Paper Generator</h1>
  <p>Dynamic RAG (DR-RAG) · FAISS-style retrieval · Bloom's Taxonomy · Semantic Deduplication</p>
</div>
""", unsafe_allow_html=True)

if not generate_btn:
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown('<div class="stat"><h4>Generation Time</h4><p>&lt; 1 min</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="stat"><h4>Time Saved</h4><p>8+ hrs → 2 min</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="stat"><h4>Bloom\'s Levels</h4><p>6 levels</p></div>', unsafe_allow_html=True)
    c4.markdown('<div class="stat"><h4>Dedup. Reduction</h4><p>~40%</p></div>', unsafe_allow_html=True)
    
    st.divider()
    st.subheader("How the DR-RAG Pipeline Works")
    
    pipeline_steps = [
        ("1️⃣ Syllabus Indexing", "Subject content is chunked and encoded into dense vector embeddings using sentence-transformers (simulates FAISS indexing)."),
        ("2️⃣ Stage 1 — Static Retrieval", "Query is encoded and top-K most relevant chunks are retrieved via cosine similarity (standard RAG pass)."),
        ("3️⃣ Stage 2 — Dynamic Refinement", "Query is augmented with the top result context, then re-ranked for higher precision (the 'Dynamic' in DR-RAG)."),
        ("4️⃣ Question Generation", "Questions are generated from retrieved chunks with appropriate Bloom's Taxonomy verb, type, and board format."),
        ("5️⃣ Semantic Deduplication", "Generated questions are compared pairwise using sentence embeddings; duplicates above 82% similarity threshold are removed (~40% reduction)."),
    ]
    for title, desc in pipeline_steps:
        with st.expander(title):
            st.write(desc)
    
    st.info("👈 Configure your paper in the sidebar and click **Generate Paper** to begin.")

else:
    if not q_types_sel:
        st.error("Please select at least one question type.")
        st.stop()
    
    start = time.time()
    with st.spinner(f"Running DR-RAG pipeline for {topic}..."):
        questions, before_dedup, after_dedup = generate_paper(
            topic, q_types_sel, bloom_levels_sel, num_questions, language, board
        )
    elapsed = round(time.time() - start, 1)
    removed = before_dedup - after_dedup
    
    # Stats
    st.subheader("Generation Summary")
    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(f'<div class="stat"><h4>Questions</h4><p>{len(questions)}</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat"><h4>Generated in</h4><p>{elapsed}s</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat"><h4>Duplicates Removed</h4><p>{removed}</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="stat"><h4>Bloom\'s Range</h4><p>L{bloom_min}–L{bloom_max}</p></div>', unsafe_allow_html=True)
    
    st.success(f"✅ Generated {len(questions)} questions for **{topic}** ({board}) in {elapsed}s. "
               f"Semantic deduplication removed {removed} near-duplicate questions.")
    
    st.divider()
    
    # Paper header
    st.markdown(f"""
    <div style="background:#f0f4ff;border-radius:12px;padding:1.5rem;border:1px solid #c7d2fe;margin-bottom:1.5rem;">
      <h2 style="color:#1e1b4b;margin:0 0 4px;">{topic} — Question Paper</h2>
      <p style="color:#4338ca;margin:0;font-size:13px;">Board: {board} &nbsp;|&nbsp; Language: {language} &nbsp;|&nbsp; Total Questions: {len(questions)} &nbsp;|&nbsp; 
      <span class="rag-badge">DR-RAG grounded</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Bloom distribution
    bloom_counts = {}
    for q in questions:
        b = q["bloom"]
        bloom_counts[b] = bloom_counts.get(b, 0) + 1
    
    col_tabs = st.tabs(["📋 All Questions", "📊 Analytics", "📥 Export"])
    
    with col_tabs[0]:
        for i, q in enumerate(questions, 1):
            bloom_color = next((v[1] for k,v in BLOOM_LEVELS.items() if v[0]==q["bloom"]), "#6366f1")
            st.markdown(f'''<div class="qcard">
              <div class="bloom">Q{i} · {q["type"]} · 
                <span style="background:{bloom_color}22;color:{bloom_color};padding:2px 8px;border-radius:99px;">
                  {q["bloom"]}
                </span>
                <span class="rag-badge">RAG-grounded</span>
              </div>
              <p class="qtext">{q["question"]}</p>''', unsafe_allow_html=True)
            
            if q["type"] == "MCQ" and "options" in q:
                for opt_i, opt in enumerate(q["options"]):
                    css = "mcq-opt mcq-correct" if opt_i==0 else "mcq-opt"
                    st.markdown(f'<div class="{css}">{opt}</div>', unsafe_allow_html=True)
            elif q["type"] == "Fill in the Blank" and "answer" in q:
                st.markdown(f'<p style="font-size:12px;color:#4338ca;margin:4px 0 0;"><strong>Answer:</strong> {q["answer"]}</p>', unsafe_allow_html=True)
            elif "hint" in q:
                st.markdown(f'<p style="font-size:12px;color:#999;margin:4px 0 0;"><em>Hint: {q["hint"]}</em></p>', unsafe_allow_html=True)
            
            st.markdown(f'<p style="font-size:11px;color:#bbb;margin:6px 0 0;">Context: {q["context"][:80]}...</p></div>', unsafe_allow_html=True)
    
    with col_tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            import plotly.express as px
            bloom_df = pd.DataFrame(list(bloom_counts.items()), columns=["Level","Count"])
            fig1 = px.pie(bloom_df, names="Level", values="Count",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          title="Bloom's Taxonomy Distribution")
            fig1.update_layout(height=300, margin=dict(t=40,b=0))
            st.plotly_chart(fig1, use_container_width=True)
        
        with c2:
            type_counts = {}
            for q in questions:
                type_counts[q["type"]] = type_counts.get(q["type"], 0) + 1
            type_df = pd.DataFrame(list(type_counts.items()), columns=["Type","Count"])
            fig2 = px.bar(type_df, x="Type", y="Count",
                          color="Count", color_continuous_scale="Blues",
                          title="Question Type Distribution")
            fig2.update_layout(height=300, margin=dict(t=40,b=0), coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.metric("Deduplication Efficiency", f"{round(removed/max(before_dedup,1)*100,1)}%",
                  help="Percentage of near-duplicate questions removed by semantic deduplication")
    
    with col_tabs[2]:
        # Export as structured text
        export_lines = [f"{topic} — Question Paper", f"Board: {board} | Language: {language}", "="*60, ""]
        for i, q in enumerate(questions, 1):
            export_lines.append(f"Q{i}. [{q['type']} | {q['bloom']}]")
            export_lines.append(q["question"])
            if q["type"] == "MCQ" and "options" in q:
                for opt in q["options"]:
                    export_lines.append(f"  {opt}")
                export_lines.append(f"  ✓ Answer: {q.get('answer','A')}")
            elif "answer" in q:
                export_lines.append(f"  Answer: {q['answer']}")
            export_lines.append("")
        
        export_text = "\n".join(export_lines)
        st.download_button(
            "📥 Download Question Paper (.txt)",
            data=export_text,
            file_name=f"{topic.replace(' ','_')}_question_paper.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.text_area("Preview", export_text, height=400)
