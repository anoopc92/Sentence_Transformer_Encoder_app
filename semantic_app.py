import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import io
from pathlib import Path

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
    .upload-section {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
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

def process_txt_file(file):
    """Process TXT file and extract text"""
    try:
        content = file.read().decode('utf-8')
        # Split by double newlines (paragraphs) or single newlines
        documents = [line.strip() for line in content.split('\n') if line.strip()]
        return documents
    except Exception as e:
        st.error(f"Error reading TXT file: {e}")
        return []

def process_csv_file(file):
    """Process CSV file and extract text from specified column"""
    try:
        df = pd.read_csv(file)
        st.write("**CSV Columns found:**", df.columns.tolist())
        
        # Let user select which column contains the text
        text_column = st.selectbox(
            "Select the column containing text documents:",
            options=df.columns.tolist()
        )
        
        if text_column:
            documents = df[text_column].dropna().astype(str).tolist()
            return documents
        return []
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return []

def process_pdf_file(file):
    """Process PDF file - requires PyPDF2"""
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        documents = []
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            # Split into sentences or paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            documents.extend(paragraphs)
        
        return documents
    except ImportError:
        st.error("PyPDF2 not installed. Please add 'PyPDF2' to requirements.txt")
        return []
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return []

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
        
        # Document statistics
        if st.session_state.documents:
            st.metric("📄 Total Documents", len(st.session_state.documents))
            st.metric("🔢 Embeddings Created", 
                     "Yes" if st.session_state.embeddings is not None else "No")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses **encoder-based models** from Hugging Face to perform:
        - 📝 Semantic Search
        - 🔄 Text Similarity
        - 📊 Document Ranking
        - 📁 File Upload Support
        """)
    
    # Main content
    if st.session_state.model is None:
        st.warning("👈 Please load a model from the sidebar to get started!")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📚 Document Upload", "🔍 Semantic Search", "📊 Similarity Analysis"])
    
    with tab1:
        st.header("Document Collection & Upload")
        
        # Upload method selection
        upload_method = st.radio(
            "Choose how to add documents:",
            ["📁 Upload Files", "✍️ Manual Input", "📋 Use Sample Documents"],
            horizontal=True
        )
        
        documents_to_add = []
        
        if upload_method == "📁 Upload Files":
            st.markdown("---")
            st.subheader("File Upload")
            
            uploaded_files = st.file_uploader(
                "Upload your documents (TXT, CSV, or PDF)",
                type=['txt', 'csv', 'pdf'],
                accept_multiple_files=True,
                help="Upload one or more files. Each file will be processed and split into documents."
            )
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.name.split('.')[-1].lower()
                    
                    st.write(f"**Processing:** {uploaded_file.name}")
                    
                    if file_type == 'txt':
                        docs = process_txt_file(uploaded_file)
                        documents_to_add.extend(docs)
                        st.success(f"✅ Extracted {len(docs)} documents from TXT file")
                    
                    elif file_type == 'csv':
                        docs = process_csv_file(uploaded_file)
                        documents_to_add.extend(docs)
                        if docs:
                            st.success(f"✅ Extracted {len(docs)} documents from CSV file")
                    
                    elif file_type == 'pdf':
                        docs = process_pdf_file(uploaded_file)
                        documents_to_add.extend(docs)
                        if docs:
                            st.success(f"✅ Extracted {len(docs)} documents from PDF file")
                
                if st.button("➕ Add These Documents"):
                    st.session_state.documents = documents_to_add
                    st.session_state.embeddings = None  # Reset embeddings
                    st.success(f"✅ Added {len(documents_to_add)} documents!")
                    st.rerun()
        
        elif upload_method == "✍️ Manual Input":
            st.markdown("---")
            st.subheader("Manual Document Input")
            
            st.info("💡 Enter your documents below, one per line. Each line will be treated as a separate document.")
            
            manual_docs = st.text_area(
                "Enter documents (one per line):",
                height=300,
                placeholder="Document 1\nDocument 2\nDocument 3\n..."
            )
            
            if manual_docs:
                documents_to_add = [doc.strip() for doc in manual_docs.split('\n') if doc.strip()]
                st.write(f"**Documents entered:** {len(documents_to_add)}")
                
                if st.button("➕ Add These Documents"):
                    st.session_state.documents = documents_to_add
                    st.session_state.embeddings = None  # Reset embeddings
                    st.success(f"✅ Added {len(documents_to_add)} documents!")
                    st.rerun()
        
        else:  # Sample Documents
            st.markdown("---")
            st.subheader("Sample Documents")
            
            sample_docs = [
                "Artificial intelligence is revolutionizing healthcare with advanced diagnostic tools and personalized treatment plans.",
                "Machine learning algorithms can analyze medical images and predict patient outcomes with remarkable accuracy.",
                "Climate change is causing rising sea levels, extreme weather events, and disrupting ecosystems worldwide.",
                "Renewable energy sources like solar and wind power are becoming more affordable and efficient every year.",
                "The stock market experienced significant volatility due to global economic uncertainty and inflation concerns.",
                "Cryptocurrency and blockchain technology are gaining adoption among institutional investors and financial institutions.",
                "Python is the most popular programming language for data science, artificial intelligence, and machine learning applications.",
                "Deep learning neural networks require large datasets and substantial computational resources for effective training.",
                "Natural language processing enables computers to understand, interpret, and generate human language effectively.",
                "Cloud computing provides scalable infrastructure and services for modern web applications and data analytics.",
                "Quantum computing promises to solve complex problems that are intractable for classical computers.",
                "Cybersecurity threats are evolving rapidly, requiring sophisticated defense mechanisms and continuous monitoring.",
                "Electric vehicles are transforming the automotive industry with improved battery technology and charging infrastructure.",
                "Space exploration has advanced with reusable rockets and plans for human missions to Mars.",
                "Biotechnology innovations are enabling gene editing, synthetic biology, and personalized medicine breakthroughs."
            ]
            
            st.write(f"**Available sample documents:** {len(sample_docs)}")
            
            with st.expander("Preview Sample Documents"):
                for i, doc in enumerate(sample_docs, 1):
                    st.text(f"{i}. {doc[:100]}...")
            
            if st.button("➕ Load Sample Documents"):
                st.session_state.documents = sample_docs
                st.session_state.embeddings = None
                st.success(f"✅ Loaded {len(sample_docs)} sample documents!")
                st.rerun()
        
        # Display current documents
        if st.session_state.documents:
            st.markdown("---")
            st.subheader("📚 Current Document Collection")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.metric("Total Documents", len(st.session_state.documents))
            
            with col2:
                if st.button("🔄 Generate Embeddings"):
                    with st.spinner("Creating embeddings... This may take a moment."):
                        st.session_state.embeddings = create_embeddings(
                            st.session_state.model,
                            st.session_state.documents
                        )
                        st.success(f"✅ Created embeddings for {len(st.session_state.documents)} documents!")
                        st.rerun()
            
            with col3:
                if st.button("🗑️ Clear All"):
                    st.session_state.documents = []
                    st.session_state.embeddings = None
                    st.rerun()
            
            # Show documents in expandable section
            with st.expander("View All Documents", expanded=False):
                for i, doc in enumerate(st.session_state.documents, 1):
                    st.text(f"{i}. {doc}")
            
            # Embeddings status
            if st.session_state.embeddings is not None:
                st.success("✅ Embeddings are ready! You can now perform semantic search.")
            else:
                st.warning("⚠️ Click 'Generate Embeddings' to enable search functionality.")
    
    with tab2:
        st.header("Semantic Search")
        
        if not st.session_state.documents:
            st.warning("⚠️ Please add documents first in the 'Document Upload' tab!")
            return
        
        if st.session_state.embeddings is None:
            st.warning("⚠️ Please generate embeddings in the 'Document Upload' tab!")
            return
        
        # Search query
        query = st.text_input(
            "🔍 Enter your search query:",
            placeholder="e.g., artificial intelligence in medicine",
            help="Enter any question or topic you want to search for in your documents"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results to show:", 1, min(20, len(st.session_state.documents)), 5)
        
        if query:
            with st.spinner("Searching..."):
                results = semantic_search(
                    query,
                    st.session_state.documents,
                    st.session_state.model,
                    st.session_state.embeddings,
                    top_k
                )
            
            st.markdown("---")
            st.subheader("🎯 Search Results")
            
            # Display results
            for i, result in enumerate(results, 1):
                score_percentage = result['score'] * 100
                
                # Color based on score
                if score_percentage >= 70:
                    color = "🟢"
                    badge_color = "success"
                elif score_percentage >= 50:
                    color = "🟡"
                    badge_color = "warning"
                else:
                    color = "🔴"
                    badge_color = "error"
                
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**{i}. {color} {result['document']}**")
                    with col2:
                        st.metric("Similarity", f"{score_percentage:.1f}%")
                    
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
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Text Similarity Analysis")
        st.write("Compare the semantic similarity between any two pieces of text.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text1 = st.text_area(
                "📝 Text 1:",
                height=150,
                placeholder="Enter first text to compare..."
            )
        
        with col2:
            text2 = st.text_area(
                "📝 Text 2:",
                height=150,
                placeholder="Enter second text to compare..."
            )
        
        if st.button("🔍 Calculate Similarity", use_container_width=True):
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
                        title={'text': "Semantic Similarity"},
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
                        st.success("🟢 **Very High Similarity** - Texts are nearly identical in meaning")
                    elif similarity_percentage >= 60:
                        st.info("🔵 **High Similarity** - Texts are closely related")
                    elif similarity_percentage >= 40:
                        st.warning("🟡 **Moderate Similarity** - Texts share some common themes")
                    else:
                        st.error("🔴 **Low Similarity** - Texts are quite different")
            else:
                st.error("⚠️ Please enter both texts to compare!")

if __name__ == "__main__":
    main()
