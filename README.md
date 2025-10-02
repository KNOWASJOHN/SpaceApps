# NASA Bioscience Explorer

A Streamlit web application for exploring and analyzing NASA's life sciences publications with AI-powered summarization capabilities.

## Features

- **Publication Database**: Browse 608 NASA life sciences publications from PubMed Central
- **Advanced Filtering**: Search and filter by research topics and organisms studied
- **Interactive Visualizations**: 
  - Topic distribution pie charts
  - Organism frequency bar charts
  - Research metrics dashboard
- **AI Summarization**: Generate summaries of scientific papers using BART model
- **Research Insights**: View most studied areas and identify research gaps

## Project Structure

```
.
├── Dockerfile              # Container configuration
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitattributes        # Git LFS configuration
├── app.py                # Main application (root)
├── src/
│   └── streamlit_app.py  # Streamlit application
└── data/
    └── SB_publication_PMC.csv  # Publications dataset
```

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/streamlit_app.py
```

### Docker Deployment

Build and run using Docker:

```bash
docker build -t nasa-bioscience-explorer .
docker run -p 8501:8501 nasa-bioscience-explorer
```

Access the app at `http://localhost:8501`

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations
- **transformers**: AI model for text summarization
- **torch**: Deep learning framework
- **trafilatura**: Web content extraction
- **altair**: Additional visualization support
- **requests**: HTTP library

## Usage

### Research Dashboard

1. Use the search bar to find publications by keywords
2. Filter by research topics (Bone Health, Muscle Physiology, etc.)
3. Filter by organisms studied (Mouse, Human, Arabidopsis, etc.)
4. View interactive charts showing research distribution
5. Browse publications with expandable details

### Paper Summarizer

1. Navigate to the "Paper Summarizer" tab
2. Enter a PubMed Central article URL
3. Click "Summarize Paper" to generate an AI summary
4. View the condensed summary of the research paper

## Data

The application uses NASA's Space Biology publications dataset containing:
- Publication titles
- PubMed Central links
- Automatically categorized topics
- Detected organism types

## Configuration

Environment variables (set in Dockerfile):
- `STREAMLIT_SERVER_PORT=8501`
- `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false`
- `HOME=/app`
- `HF_HOME=/app/.cache/huggingface`
- `TRANSFORMERS_CACHE=/app/.cache/huggingface`

## Known Limitations

- Summarization requires internet connection for model downloads
- AI summaries may take time to generate on first use
- Limited to English language publications
- Requires sufficient memory for transformer models (~2GB)
