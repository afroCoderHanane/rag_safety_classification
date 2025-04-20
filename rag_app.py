import streamlit as st
import pandas as pd
import os
import tempfile
import json
from datetime import datetime
import time
import traceback

# Import Google Cloud and LangChain packages
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from langchain_community.vectorstores import Chroma
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.schema import Document

# Import Google Generative AI SDK
from google import genai
from google.genai.types import HttpOptions

# Set page configuration
st.set_page_config(
    page_title="RAG-based Hate Speech Classification",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state for storing the dataset and vector store
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "project_id" not in st.session_state:
    st.session_state.project_id = None
if "location" not in st.session_state:
    st.session_state.location = None
if "model_name" not in st.session_state:
    st.session_state.model_name = None
if "api_initialized" not in st.session_state:
    st.session_state.api_initialized = False
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "genai_client" not in st.session_state:
    st.session_state.genai_client = None


def setup_authentication():
    """Helper function to set up authentication for Vertex AI"""
    import os
    
    # Display authentication status
    st.sidebar.subheader("Authentication Status")
    
    # Check if GOOGLE_APPLICATION_CREDENTIALS is set
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    if creds_path:
        st.sidebar.success(f"✅ Found credentials at: {creds_path}")
    else:
        st.sidebar.warning("⚠️ GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        
        # Add service account key uploader
        st.sidebar.subheader("Upload Service Account Key")
        uploaded_key = st.sidebar.file_uploader("Service Account JSON Key", type=["json"])
        
        if uploaded_key is not None:
            # Save the uploaded key to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                tmp_file.write(uploaded_key.getvalue())
                key_path = tmp_file.name
            
            # Set environment variable
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
            st.sidebar.success(f"✅ Credentials set from uploaded key: {key_path}")
            
            # Add a rerun button to apply the credentials
            if st.sidebar.button("Apply Credentials"):
                st.experimental_rerun()
        
        # Add instructions for authentication
        with st.sidebar.expander("Authentication Instructions"):
            st.markdown("""
            ### How to Authenticate with Google Cloud
            
            **Option 1: Using gcloud CLI (Recommended)**
            
            1. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
            2. Open a terminal and run:
            ```
            gcloud auth application-default login
            ```
            3. Follow the browser prompts to log in
            
            **Option 2: Using a Service Account Key**
            
            1. In Google Cloud Console, go to IAM & Admin > Service Accounts
            2. Create a service account or select an existing one
            3. Add the "Vertex AI User" role
            4. Create and download a JSON key
            5. Upload the key using the file uploader above
            
            **Option 3: Set Environment Variable**
            
            1. Set the GOOGLE_APPLICATION_CREDENTIALS environment variable:
            ```
            # Windows
            set GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
            
            # macOS/Linux
            export GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json
            ```
            2. Restart this application
            """)
    
    # Return True if authenticated, False otherwise
    return creds_path is not None


def initialize_vertex_ai(project_id, location):
    """Initialize Vertex AI API and GenAI Client"""
    try:
        # Set project and location
        vertexai.init(project=project_id, location=location)
        
        # Set environment variables for GenAI SDK
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = location
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
        
        # Initialize the GenAI client
        try:
            client = genai.Client(http_options=HttpOptions(api_version="v1"))
            
            # Test the client with a simple request
            response = client.models.generate_content(
                model="gemini-2.0-flash-001",
                contents="Respond with only one word: Hello",
            )
            
            if response and hasattr(response, 'text'):
                st.success(f"✅ Successfully initialized Gemini 2.0 Flash model")
                st.session_state.genai_client = client
                st.session_state.api_initialized = True
                st.session_state.project_id = project_id
                st.session_state.location = location
                st.session_state.model_name = "gemini-2.0-flash-001"
                return True
            else:
                st.error("⚠️ Gemini model returned unexpected response format")
                return False
                
        except Exception as model_error:
            st.error(f"Error initializing Gemini model: {str(model_error)}")
            st.error("""
            There was an error accessing the Gemini model. This could be due to:
            1. Insufficient permissions for your account
            2. The Vertex AI API not being fully enabled
            3. Billing not being set up for the project
            4. The model not being available in the selected region
            
            Please check these settings in your Google Cloud Console.
            """)
            st.session_state.api_initialized = False
            return False
            
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")
        st.error("""
        Authentication failed. Please check:
        1. Your Google Cloud Project ID is correct
        2. You have authenticated with gcloud CLI using: gcloud auth application-default login
        3. The Vertex AI API is enabled in your project
        4. You have the necessary permissions (at least Vertex AI User role)
        """)
        st.session_state.api_initialized = False
        return False


def load_dataset(uploaded_file):
    """Load dataset from CSV file"""
    try:
        # Create a temporary file to store the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Read the dataset
        df = pd.read_csv(tmp_file_path)
        
        # Check if required columns exist
        required_columns = ["prompt", "label"]
        if not all(col in df.columns for col in required_columns):
            st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
            return None
        
        # Remove temporary file
        os.unlink(tmp_file_path)
        
        return df
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def create_vector_store(dataset):
    """Create vector store from dataset using Vertex AI embeddings"""
    with st.spinner("Creating embeddings and vector store... This may take a while depending on the dataset size."):
        try:
            # Check if project_id and location are set
            if not st.session_state.project_id or not st.session_state.location:
                st.error("Project ID or location not set. Please initialize Vertex AI API first.")
                return None
                
            # Initialize embedding model with detailed error handling
            try:
                embedding_model = VertexAIEmbeddings(
                    model_name="text-embedding-005",
                    project=st.session_state.project_id,
                    location=st.session_state.location,
                )
                
                # Test the embedding model with a simple example
                test_embedding = embedding_model.embed_query("test query")
                if not test_embedding or len(test_embedding) == 0:
                    st.error("Embedding model returned empty embedding. Authentication may be incorrect.")
                    return None
                    
                st.success("✅ Successfully connected to Vertex AI Embeddings API")
                
            except Exception as embed_error:
                st.error(f"Error initializing embedding model: {str(embed_error)}")
                st.info("""
                ### Embedding Model Error
                
                This usually means one of the following:
                1. The Vertex AI API is not enabled for your project
                2. Your authentication credentials are incorrect
                3. Your account doesn't have permission to use Vertex AI Embeddings
                4. You haven't completed the billing setup for your Google Cloud project
                
                Check these settings in your Google Cloud Console.
                """)
                return None
            
            # Create documents for vector store
            st.info("Creating document embeddings for the dataset...")
            
            documents = []
            progress_bar = st.progress(0)
            
            for idx, row in dataset.iterrows():
                content = row["prompt"]
                metadata = {
                    "label": row["label"],
                    "index": idx
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
                # Update progress
                progress = (idx + 1) / len(dataset)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            
            # Create vector store
            st.info("Creating vector store from embeddings...")
            
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embedding_model,
                persist_directory=None,  # In-memory store for simplicity
            )
            
            return vector_store
        
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None


def get_similar_prompts(vector_store, query_text, top_k=5):
    """Get similar prompts from vector store"""
    try:
        # Check if vector_store is None
        if vector_store is None:
            st.error("Vector store is not initialized. Please create the vector store first.")
            return []
            
        # Search vector store for similar prompts
        results = vector_store.similarity_search_with_score(
            query=query_text,
            k=top_k
        )
        
        similar_examples = []
        for doc, score in results:
            similar_examples.append({
                "prompt": doc.page_content,
                "label": doc.metadata["label"],
                "similarity_score": score,
                "index": doc.metadata["index"]
            })
        
        # Check if we found any examples
        if not similar_examples:
            st.warning(f"No similar examples found for: '{query_text[:50]}...'")
        
        return similar_examples
    
    except Exception as e:
        st.error(f"Error retrieving similar prompts: {str(e)}")
        st.error(f"Debug info - Vector store type: {type(vector_store)}")
        return []


def classify_prompt_with_rag(prompt, similar_examples):
    """Classify prompt using RAG and Gemini 2.0 Flash with improved instructions"""
    try:
        # Check if similar_examples is None or empty
        if not similar_examples:
            st.warning("No similar examples found. Using direct classification without RAG context.")
            similar_examples = []
            context = "No similar examples found. Please classify based on the guidelines below.\n\n"
        else:
            # Create context from similar examples
            context = "Here are similar examples with their classifications:\n\n"
            for i, example in enumerate(similar_examples):
                context += f"Example {i+1}: \"{example['prompt']}\"\n"
                context += f"Classification: {example['label']}\n\n"
        
        # Create prompt with RAG context and better classification guidelines
        rag_prompt = f"""{context}
Your task is to classify the following text as either "hate_speech" or "not_hate_speech".

Guidelines for classification:
1. Hate speech: Content that expresses, incites, or promotes hatred, violence, or prejudicial stereotypes against protected groups based on attributes such as race, ethnicity, gender, religion, sexual orientation, or disability.
2. Not hate speech: Content that may be critical or negative but doesn't target protected groups or express/promote hatred, violence, or discrimination.

Analyze the text carefully and avoid overgeneralizing. A text isn't hate speech simply because it contains criticism, strong language, or controversial opinions if it doesn't target or demean protected groups.

Text to classify: "{prompt}"

Respond with ONLY one of these two labels:
- hate_speech
- not_hate_speech
"""
        
        # Use the Gemini client to generate the classification
        try:
            client = st.session_state.genai_client
            if client is None:
                st.error("Gemini client not initialized. Please initialize Vertex AI API first.")
                return "Error: Gemini client not initialized"
                
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=rag_prompt
            )
            
            # Extract and normalize classification
            result = response.text.strip().lower()
            
            # Ensure only valid labels are returned
            # if "hate_speech" in result:
            #     return "hate_speech"
            # elif "not_hate_speech" in result or "not hate_speech" in result:
            #     return "not_hate_speech"
            # else:
                # Default to the model's raw output if parsing fails
            return result
                
        except Exception as model_error:
            st.error(f"Error with Gemini model: {str(model_error)}")
            st.error("This could be due to authentication issues or lack of permissions to use the model.")
            return "Error: Model access failed"
    
    except Exception as e:
        st.error(f"Error classifying prompt: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return "Error: Classification failed"


def batch_classify(dataset, vector_store, sample_size=None):
    """Perform batch classification on dataset"""
    try:
        results = []
        
        # Sample the dataset if requested
        if sample_size and sample_size < len(dataset):
            processing_dataset = dataset.sample(sample_size)
        else:
            processing_dataset = dataset
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each prompt
        for idx, (_, row) in enumerate(processing_dataset.iterrows()):
            prompt = row["prompt"]
            true_label = row["label"]
            
            # Update progress
            progress = (idx + 1) / len(processing_dataset)
            progress_bar.progress(progress)
            status_text.text(f"Processing {idx+1}/{len(processing_dataset)}: {prompt[:50]}...")
            
            # Get similar examples
            similar_examples = get_similar_prompts(vector_store, prompt)
            
            # Check if similar_examples is None or empty
            if similar_examples is None:
                similar_examples = []
            
            # Classify with RAG
            predicted_label = classify_prompt_with_rag(prompt, similar_examples)
            
            # Store result
            result = {
                "prompt": prompt,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "similar_examples": similar_examples
            }
            results.append(result)
            
            # Add a small delay to prevent rate limiting
            time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return results
    
    except Exception as e:
        st.error(f"Error in batch classification: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return []


def export_results(results):
    """Export results to JSON file"""
    if not results:
        st.error("No results to export!")
        return None
    
    try:
        # Convert results to JSON
        results_json = json.dumps(results, indent=2)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"classification_results_{timestamp}.json"
        
        return results_json, filename
    
    except Exception as e:
        st.error(f"Error exporting results: {str(e)}")
        return None


def calculate_metrics(results):
    """Calculate classification metrics"""
    if not results:
        return None
    
    try:
        total = len(results)
        correct = sum(1 for r in results if r["true_label"] == r["predicted_label"])
        accuracy = correct / total if total > 0 else 0
        
        # Count true/false positives/negatives
        tp = sum(1 for r in results if r["true_label"] == "hate_speech" and r["predicted_label"] == "hate_speech")
        fp = sum(1 for r in results if r["true_label"] == "not_hate_speech" and r["predicted_label"] == "hate_speech")
        tn = sum(1 for r in results if r["true_label"] == "not_hate_speech" and r["predicted_label"] == "not_hate_speech")
        fn = sum(1 for r in results if r["true_label"] == "hate_speech" and r["predicted_label"] == "not_hate_speech")
        
        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "correct_predictions": correct,
            "total_samples": total
        }
    
    except Exception as e:
        st.error(f"Error calculating metrics: {str(e)}")
        return None


# Main App UI
st.title("🔍 RAG-based Hate Speech Classification")
st.markdown("""
This app uses Retrieval Augmented Generation (RAG) with Gemini 2.0 Flash to classify text as hate speech or not hate speech.
It retrieves similar examples from your dataset to provide context for the classification model.
""")

# Sidebar for configuration
st.sidebar.title("Configuration")

# Authentication setup
auth_status = setup_authentication()

# Google Cloud Configuration
st.sidebar.header("Google Cloud Setup")
project_id = st.sidebar.text_input("Google Cloud Project ID", key="project_id_input")
location = st.sidebar.selectbox(
    "Vertex AI Location",
    ["us-central1", "us-east1", "us-west1", "europe-west4", "asia-east1"],
    index=0,
    key="location_input"
)

st.sidebar.info("This app uses the Gemini 2.0 Flash model for classification.")

# Initialize API button
if st.sidebar.button("Initialize Vertex AI API"):
    if project_id:
        initialize_vertex_ai(project_id, location)
    else:
        st.sidebar.error("Please enter your Google Cloud Project ID")

# Dataset Upload
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload hate speech dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    dataset = load_dataset(uploaded_file)
    if dataset is not None:
        st.session_state.dataset = dataset
        st.sidebar.success(f"✅ Dataset loaded successfully! ({len(dataset)} rows)")
        
        # Display dataset sample in sidebar
        with st.sidebar.expander("View Dataset Sample"):
            st.dataframe(dataset.head())
        
        # Create vector store
        if st.sidebar.button("Create Vector Store"):
            if st.session_state.api_initialized:
                vector_store = create_vector_store(dataset)
                if vector_store is not None:
                    st.session_state.vector_store = vector_store
                    st.sidebar.success("✅ Vector store created successfully!")
            else:
                st.sidebar.error("Please initialize Vertex AI API first")

# Main content tabs
tab1, tab2, tab3 = st.tabs(["Single Prompt Classification", "Batch Classification", "Results Analysis"])

with tab1:
    st.header("Single Prompt Classification")
    st.markdown("Enter a text prompt to classify it as hate speech or not hate speech.")
    
    user_prompt = st.text_area("Enter text to classify:", height=100)
    
    if st.button("Classify"):
        if not st.session_state.api_initialized:
            st.error("Please initialize Vertex AI API first!")
            st.info("Click the 'Initialize Vertex AI API' button in the sidebar after entering your project ID and selecting a location.")
        elif st.session_state.vector_store is None:
            st.error("Please create the vector store first!")
            st.info("Upload a dataset and click the 'Create Vector Store' button in the sidebar.")
        elif user_prompt:
            try:
                with st.spinner("Classifying..."):
                    # Get similar examples
                    similar_examples = get_similar_prompts(st.session_state.vector_store, user_prompt)
                    
                    if similar_examples is None:
                        st.error("Failed to retrieve similar examples. Vector store may not be properly initialized.")
                        similar_examples = []
                    
                    # Classify with RAG
                    result = classify_prompt_with_rag(user_prompt, similar_examples)
                    
                    # Check if classification was successful
                    if result and not result.startswith("Error:"):
                        # Display result
                        st.subheader("Classification Result:")
                        if result == "hate_speech":
                            st.error(f"Result: {result}")
                        else:
                            st.success(f"Result: {result}")
                        
                        # Display similar examples
                        if similar_examples:
                            st.subheader("Similar Examples Used for Context:")
                            for i, example in enumerate(similar_examples):
                                with st.expander(f"Example {i+1} - Similarity: {1 - example['similarity_score']:.4f}"):
                                    st.markdown(f"**Text:** {example['prompt']}")
                                    if example['label'] == "hate_speech":
                                        st.markdown(f"**Label:** 🚫 {example['label']}")
                                    else:
                                        st.markdown(f"**Label:** ✅ {example['label']}")
                        else:
                            st.info("No similar examples were found in the dataset. Classification was performed without RAG context.")
                    else:
                        st.error(f"Classification failed: {result}")
                        st.info("Check the error messages and make sure your authentication and API access are set up correctly.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.error("Debug information:")
                st.error(f"- API initialized: {st.session_state.api_initialized}")
                st.error(f"- Vector store type: {type(st.session_state.vector_store)}")
                st.error(f"- GenAI client initialized: {st.session_state.genai_client is not None}")
        else:
            st.warning("Please enter a text prompt to classify!")

with tab2:
    st.header("Batch Classification")
    st.markdown("Classify multiple examples from your dataset at once.")
    
    # Batch size selection
    sample_size = st.number_input(
        "Number of samples to process (0 for all):",
        min_value=0,
        max_value=None if st.session_state.dataset is None else len(st.session_state.dataset),
        value=10
    )
    
    if st.button("Run Batch Classification"):
        if not st.session_state.api_initialized:
            st.error("Please initialize Vertex AI API first!")
        elif st.session_state.vector_store is None:
            st.error("Please create the vector store first!")
        elif st.session_state.dataset is None:
            st.error("Please upload a dataset first!")
        else:
            # Run batch classification
            with st.spinner("Running batch classification... This may take a while."):
                results = batch_classify(
                    st.session_state.dataset,
                    st.session_state.vector_store,
                    sample_size if sample_size > 0 else None
                )
                
                if results:
                    st.session_state.batch_results = results
                    st.success(f"✅ Batch classification completed for {len(results)} samples!")
                    
                    # Display download button
                    export_data = export_results(results)
                    if export_data:
                        results_json, filename = export_data
                        st.download_button(
                            label="Download Results as JSON",
                            data=results_json,
                            file_name=filename,
                            mime="application/json"
                        )

with tab3:
    st.header("Results Analysis")
    
    if st.session_state.batch_results:
        results = st.session_state.batch_results
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        if metrics:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision", f"{metrics['precision']:.2%}")
            col3.metric("Recall", f"{metrics['recall']:.2%}")
            col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            
            st.info(f"Correctly classified {metrics['correct_predictions']} out of {metrics['total_samples']} samples.")
            
            # Display results table
            st.subheader("Classification Results")
            
            results_df = pd.DataFrame([
                {
                    "Prompt": r["prompt"][:50] + "..." if len(r["prompt"]) > 50 else r["prompt"],
                    "True Label": r["true_label"],
                    "Predicted Label": r["predicted_label"],
                    "Correct": r["true_label"] == r["predicted_label"]
                }
                for r in results
            ])
            
            # Add styling to the dataframe
            def highlight_correct(val):
                return 'background-color: #CCFFCC' if val else 'background-color: #FFCCCC'
            
            styled_df = results_df.style.apply(
                lambda x: [highlight_correct(val) for val in x], 
                subset=['Correct']
            )
            
            st.dataframe(styled_df)
            
            # Error analysis
            st.subheader("Error Analysis")
            
            # Filter incorrect predictions
            incorrect = [r for r in results if r["true_label"] != r["predicted_label"]]
            
            if incorrect:
                st.markdown(f"Found {len(incorrect)} incorrect predictions.")
                
                for i, case in enumerate(incorrect[:10]):  # Limit to first 10 errors
                    with st.expander(f"Error Case {i+1}: Predicted '{case['predicted_label']}' but was '{case['true_label']}'"):
                        st.markdown(f"**Prompt:** {case['prompt']}")
                        st.markdown("**Similar examples used for context:**")
                        
                        for j, ex in enumerate(case['similar_examples']):
                            st.markdown(f"- Example {j+1}: \"{ex['prompt'][:100]}...\" (Label: {ex['label']})")
            else:
                st.success("No incorrect predictions found!")
    else:
        st.info("Run a batch classification first to see results analysis.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    "This app uses Google Cloud Vertex AI with Gemini 2.0 Flash and RAG to classify text as hate speech or not hate speech."
)