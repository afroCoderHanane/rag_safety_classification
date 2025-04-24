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
git clone https://github.com/afroCoderHanane/rag-hate-speech-classification.git
cd rag-hate-speech-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up Google Cloud authentication:
```bash
gcloud auth application-default login
```

## Docker Installation

You can also run the application and dataset generator using Docker.

### Building the Docker Image

```bash
docker build -t rag-classifier .
```

### Running the Docker Container

#### 1. Running the Streamlit App

```bash
docker run -p 8501:8501 -v $(pwd):/data rag-classifier app
```

Access the app in your browser at: http://localhost:8501

#### 2. Generating a Hate Speech Dataset

```bash
docker run -v $(pwd):/data rag-classifier generate --samples=200 --output=/data/my_dataset.csv
```

Options:
- `--samples=NUMBER`: Number of examples to generate (default: 200)
- `--output=PATH`: Where to save the dataset (default: timestamped filename in container)

### Using Docker Compose

1. Start the application:
```bash
docker-compose up rag-app
```

2. Generate a dataset:
```bash
docker-compose run generator
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