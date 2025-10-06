import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

# Set page configuration
st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'documents' not in st.session_state:
    st.session_state.documents = []

@st.cache_resource
def load_model(model_name):
    """Load the sentence transformer model"""
    return SentenceTransformer(model_name)

def create_embeddings(model, texts):
    """Create embeddings for given texts"""
    return model.encode(texts, convert_to_tensor=True)

def semantic_search(query, documents, model, embeddings, top_k=5):
    """Perform semantic search"""
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    
    # Get top k results
    top_results = torch.topk(cos_scores, k=min(top_k, len(documents)))
    
    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            'document': documents[idx],
            'score': score.item()
        })
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Semantic Search Engine</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Sentence-BERT Encoder Model")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "all-MiniLM-L6-v2 (Fast)": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2 (Accurate)": "sentence-transformers/all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1 (Q&A)": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
        }
        
        selected_model = st.selectbox(
            "Choose Model",
            options=list(model_options.keys())
        )
        
        model_name = model_options[selected_model]
        
        st.info(f"**Model:** {model_name}\n\n"
                f"**Type:** Encoder-based transformer\n\n"
                f"**Use Case:** Semantic similarity & search")
        
        # Load model button
        if st.button("🚀 Load Model"):
            with st.spinner("Loading model..."):
                st.session_state.model = load_model(model_name)
                st.success("✅ Model loaded successfully!")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses **encoder-based models** from Hugging Face to perform:
        - 📝 Semantic Search
        - 🔄 Text Similarity
        - 📊 Document Ranking
        """)
    
    # Main content
    if st.session_state.model is None:
        st.warning("👈 Please load a model from the sidebar to get started!")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Document Collection", "🔍 Semantic Search", "📊 Similarity Analysis"])
    
    with tab1:
        st.header("Document Collection")
        
        # Sample documents
        sample_docs = [
            "Artificial intelligence is revolutionizing healthcare with advanced diagnostic tools.",
            "Machine learning algorithms can predict patient outcomes with high accuracy.",
            "Climate change is causing rising sea levels and extreme weather events.",
            "Renewable energy sources like solar and wind are becoming more affordable.",
            "The stock market showed volatility due to global economic uncertainty.",
            "Cryptocurrency adoption is growing among institutional investors.",
            "Python is a popular programming language for data science and AI.",
            "Deep learning models require large amounts of training data.",
            "Natural language processing enables computers to understand human language.",
            "Cloud computing provides scalable infrastructure for modern applications."
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_sample = st.checkbox("Use sample documents", value=True)
            
            if use_sample:
                st.session_state.documents = sample_docs
            else:
                custom_docs = st.text_area(
                    "Enter your documents (one per line)",
                    height=300,
                    placeholder="Enter documents here, separated by new lines..."
                )
                
                if custom_docs:
                    st.session_state.documents = [doc.strip() for doc in custom_docs.split('\n') if doc.strip()]
        
        with col2:
            st.metric("Total Documents", len(st.session_state.documents))
            
            if st.button("🔄 Generate Embeddings"):
                if st.session_state.documents:
                    with st.spinner("Creating embeddings..."):
                        st.session_state.embeddings = create_embeddings(
                            st.session_state.model,
                            st.session_state.documents
                        )
                        st.success(f"✅ Created {len(st.session_state.documents)} embeddings!")
                else:
                    st.error("Please add documents first!")
        
        # Display documents
        if st.session_state.documents:
            st.subheader("Current Documents")
            for i, doc in enumerate(st.session_state.documents, 1):
                st.text(f"{i}. {doc}")
    
    with tab2:
        st.header("Semantic Search")
        
        if st.session_state.embeddings is None:
            st.warning("⚠️ Please generate embeddings first in the Document Collection tab!")
            return
        
        # Search query
        query = st.text_input(
            "Enter your search query",
            placeholder="e.g., artificial intelligence in medicine"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", 1, 10, 5)
        
        if query:
            with st.spinner("Searching..."):
                results = semantic_search(
                    query,
                    st.session_state.documents,
                    st.session_state.model,
                    st.session_state.embeddings,
                    top_k
                )
            
            st.subheader("🎯 Search Results")
            
            # Display results
            for i, result in enumerate(results, 1):
                score_percentage = result['score'] * 100
                
                # Color based on score
                if score_percentage >= 70:
                    color = "🟢"
                elif score_percentage >= 50:
                    color = "🟡"
                else:
                    color = "🔴"
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{i}. {color} {result['document']}**")
                    with col2:
                        st.metric("Score", f"{score_percentage:.1f}%")
                    st.progress(result['score'])
                    st.markdown("---")
            
            # Visualization
            st.subheader("📊 Score Distribution")
            fig = go.Figure(data=[
                go.Bar(
                    x=[f"Result {i+1}" for i in range(len(results))],
                    y=[r['score'] * 100 for r in results],
                    marker_color='rgb(102, 126, 234)',
                    text=[f"{r['score']*100:.1f}%" for r in results],
                    textposition='auto',
                )
            ])
            fig.update_layout(
                yaxis_title="Similarity Score (%)",
                xaxis_title="Search Results",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Text Similarity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "Text 1",
                height=150,
                placeholder="Enter first text..."
            )
        
        with col2:
            text2 = st.text_area(
                "Text 2",
                height=150,
                placeholder="Enter second text..."
            )
        
        if st.button("Calculate Similarity"):
            if text1 and text2:
                with st.spinner("Computing similarity..."):
                    embedding1 = st.session_state.model.encode(text1, convert_to_tensor=True)
                    embedding2 = st.session_state.model.encode(text2, convert_to_tensor=True)
                    
                    similarity = util.cos_sim(embedding1, embedding2).item()
                    similarity_percentage = similarity * 100
                
                # Display result
                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.metric(
                        "Similarity Score",
                        f"{similarity_percentage:.2f}%",
                        delta=None
                    )
                    
                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=similarity_percentage,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Similarity"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 33], 'color': "lightgray"},
                                {'range': [33, 66], 'color': "gray"},
                                {'range': [66, 100], 'color': "lightblue"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretation
                    if similarity_percentage >= 80:
                        st.success("🟢 Very High Similarity - Texts are nearly identical in meaning")
                    elif similarity_percentage >= 60:
                        st.info("🔵 High Similarity - Texts are closely related")
                    elif similarity_percentage >= 40:
                        st.warning("🟡 Moderate Similarity - Texts share some common themes")
                    else:
                        st.error("🔴 Low Similarity - Texts are quite different")
            else:
                st.error("Please enter both texts!")

if __name__ == "__main__":
    main()
