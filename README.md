# RAG-based Hate Speech Classification Application

This Streamlit application uses Retrieval Augmented Generation (RAG) to classify text as hate speech or not hate speech, leveraging Google Cloud Vertex AI's embedding and generative models.

## Features

- **RAG-based Classification**: Retrieves similar examples from your dataset to provide context for more accurate classification
- **Single Prompt Classification**: Classify individual text inputs
- **Batch Processing**: Process multiple examples at once for evaluation
- **Results Analysis**: View performance metrics and analyze classification errors
- **Context Visualization**: See which similar examples influenced each classification

## Prerequisites

1. A Google Cloud account with Vertex AI API enabled
2. A CSV dataset with hate speech examples (must contain "prompt" and "label" columns)
3. Python 3.8 or higher

## Installation

1. Clone this repository:
```bash
git clone https://github.com/afroCoderHanane/rag_safety_classification.git
cd rag_safety_classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud authentication:
```bash
gcloud auth application-default login
```

## Dataset Format

Your dataset should be a CSV file with at least these columns:
- `prompt`: The text to classify
- `label`: The classification label (should be either "hate_speech" or "not_hate_speech")

Example:
```
prompt,label
"I hate all people from that country",hate_speech
"I love sunny days at the beach",not_hate_speech
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. In your web browser, you'll see the application interface.

3. In the sidebar:
   - Enter your Google Cloud Project ID
   - Select a Vertex AI Location
   - Click "Initialize Vertex AI API"
   - Upload your hate speech dataset
   - Click "Create Vector Store" to create embeddings

4. Use the tabs to perform single classifications or batch processing

## How It Works

RAG is a method that combines retrieval and generation to deliver more accurate and context-aware results. The retrieval process pulls relevant documents from a knowledge base, while the generation process uses a language model to create a coherent response based on the retrieved content.

This application:
1. Creates embeddings for all examples in your hate speech dataset
2. Stores these embeddings in a vector store
3. When classifying new text:
   - Creates an embedding for the new text
   - Finds the 5 most similar examples from your dataset
   - Sends these similar examples as context to the Vertex AI model
   - Uses the model to classify the new text as hate speech or not

## Tips for Better Results

- Ensure your dataset has diverse examples
- Use a large enough dataset for better retrieval (ideally 1000+ examples)
- Experiment with different Vertex AI models
- Adjust the number of similar examples retrieved (default is 5)

## License

MIT