#!/bin/bash

# Function to print help message
function print_help {
    echo "Usage: docker run [options] [command]"
    echo ""
    echo "Commands:"
    echo "  app                  Run the Streamlit RAG classification app (default)"
    echo "  generate [options]   Generate a hate speech dataset"
    echo ""
    echo "Options for 'generate':"
    echo "  --samples=NUMBER     Number of samples to generate (default: 200)"
    echo "  --output=FILENAME    Output filename (default: hate_speech_dataset_TIMESTAMP.csv)"
    echo ""
}

# Default command is to run the Streamlit app
COMMAND=${1:-app}

case "$COMMAND" in
    app)
        echo "Starting RAG Classification Streamlit app..."
        # Use PORT environment variable from Cloud Run, default to 8501 if not set
        export PORT=${PORT:-8501}
        streamlit run rag_app.py --server.port=$PORT --server.address=0.0.0.0
        ;;
        
    generate)
        echo "Generating hate speech dataset..."
        
        # Parse arguments
        SAMPLES=200
        OUTPUT=""
        
        for arg in "${@:2}"; do
            case "$arg" in
                --samples=*)
                    SAMPLES="${arg#*=}"
                    ;;
                --output=*)
                    OUTPUT="${arg#*=}"
                    ;;
                *)
                    echo "Unknown argument: $arg"
                    print_help
                    exit 1
                    ;;
            esac
        done
        
        # Build the command
        CMD="python dataset_generator.py --samples $SAMPLES"
        if [ ! -z "$OUTPUT" ]; then
            CMD="$CMD --output $OUTPUT"
        fi
        
        # Run the command
        echo "Running: $CMD"
        eval $CMD
        ;;
        
    help)
        print_help
        ;;
        
    *)
        echo "Unknown command: $COMMAND"
        print_help
        exit 1
        ;;
esac