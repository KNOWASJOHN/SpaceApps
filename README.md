# 🚀 NASA Bioscience Explorer

A Streamlit web application for exploring and analyzing NASA life sciences publications with AI-powered summarization capabilities.

![NASA Bioscience Explorer](https://github.com/KNOWASJOHN/SpaceApps/blob/main/dashboard.png?raw=true)

## 📋 Overview

NASA Bioscience Explorer provides an interactive platform to explore 608 NASA life sciences publications. The application features advanced filtering, visualization, and AI-powered section-by-section paper summarization to help researchers quickly understand key findings.

## ✨ Features

### 🔍 Search & Filter
- **Keyword Search**: Search across titles, topics, and organisms
- **Topic Filtering**: Filter by research areas (Bone Health, Muscle Physiology, Immune System, etc.)
- **Organism Filtering**: Filter by studied organisms (Mouse, Human, Arabidopsis, etc.)
- **Real-time Filtering**: Instant results as you type and select filters

### 📊 Research Dashboard
- **Interactive Visualizations**: Pie charts and bar graphs showing research distribution
- **Publication Metrics**: Key statistics on publications, topics, and organisms
- **Publication Browser**: Browse all filtered publications with direct paper links
- **Research Insights**: Identify most studied areas and research gaps

### 📄 AI Paper Summarizer
- **Section-wise Summarization**: AI-generated summaries for specific paper sections
- **Smart Content Extraction**: Automatically extracts Introduction, Methods, Results, Discussion, and Conclusion sections
- **Multi-model Support**: Uses Facebook's BART-large-CNN model for high-quality summarization
- **URL-based Processing**: Summarize any scientific paper by providing its URL

### 🤖 AI-Powered Features
- **Section Detection**: Intelligent identification of academic paper sections
- **Chunk Processing**: Handles long documents through smart text chunking
- **Cached Results**: Efficient caching for faster repeated access
- **Fallback Mechanisms**: Robust error handling with fallback options

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/nasa-bioscience-explorer.git
   cd nasa-bioscience-explorer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up data directory**
   ```bash
   mkdir data
   # Place your SB_publication_PMC.csv file in the data directory
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure

```
nasa-bioscience-explorer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── data/                 # Data directory
│   └── SB_publication_PMC.csv  # Publication dataset
```

## 📊 Data Format

The application expects a CSV file with the following columns:
- `Title`: Publication title
- `Link`: URL to the publication
- Additional columns for metadata

## 🔧 Configuration

### Environment Variables
No environment variables required. All configuration is handled within the application.

### Model Configuration
- **Summarization Model**: `facebook/bart-large-cnn`
- **Text Chunk Size**: 1000 characters
- **Max Input Length**: 8000 characters
- **Section Limit**: 2000 characters per section

## 🚀 Usage

### Research Dashboard
1. Use the search bar and filters to find relevant publications
2. View distribution charts to understand research trends
3. Click on individual publications to generate AI summaries
4. Explore research insights and identified gaps

### Paper Summarizer
1. Navigate to the "Paper Summarizer" tab
2. Enter any scientific paper URL (PMC, PubMed, etc.)
3. Click "Summarize Paper" to generate section-wise summaries
4. View summaries for Introduction, Methods, Results, Discussion, and Conclusion

## 🎯 Supported Research Areas

- **Bone Health**: Bone density, skeletal research, osteoporosis
- **Muscle Physiology**: Muscle atrophy, physiology studies
- **Immune System**: Immune response, infections, microbiome
- **Plant Biology**: Arabidopsis, plant growth, root systems
- **Radiation Effects**: DNA damage, genomic stability
- **Microgravity Adaptation**: Gravity effects, space adaptation
- **Other**: Miscellaneous life sciences research

## 🧬 Supported Organisms

- Mouse
- Human/Astronaut
- Arabidopsis
- Drosophila
- Rat
- Various (multiple organisms)

## 🔍 API & Libraries Used

- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Transformers**: Hugging Face AI models (BART)
- **Trafilatura**: Web content extraction
- **Pandas**: Data manipulation and analysis
- **Textwrap**: Text formatting utilities
