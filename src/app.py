import streamlit as st
import pandas as pd
import plotly.express as px
import trafilatura
from transformers import pipeline
import textwrap

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="NASA Bioscience Explorer",
    page_icon="https://github.com/KNOWASJOHN/SpaceApps/blob/main/logo_.png?raw=true",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide the sidebar by default
)

# Now import other libraries
from transformers import pipeline
import textwrap

# Cache the summarizer so it only loads once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Cached summarization function
def summarize_entire_text(text):
    summarizer = load_summarizer()
    
    # Limit input to 8000 characters
    text = text[:8000] if len(text) > 8000 else text
    
    # Split text into chunks of 4000 characters
    chunk_size = 4000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        if chunk.strip():  # Only process non-empty chunks
            result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            summaries.append(result[0]['summary_text'])
    
    # Combine summaries if there are multiple chunks
    if len(summaries) > 1:
        combined_summary = " ".join(summaries)
        # Final summarization pass to consolidate multiple summaries
        final_result = summarizer(combined_summary, max_length=250, min_length=100, do_sample=False)
        return final_result[0]['summary_text']
    elif len(summaries) == 1:
        return summaries[0]
    else:
        return "No text to summarize."

# Load data with caching
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/SB_publication_PMC.csv')
        if df.empty:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()
    
    # Simple categorization based on title keywords
    def categorize_topic(title):
        title_lower = title.lower()
        if any(word in title_lower for word in ['bone', 'skeletal', 'oste']):
            return 'Bone Health'
        elif any(word in title_lower for word in ['muscle', 'atrophy']):
            return 'Muscle Physiology'
        elif any(word in title_lower for word in ['immune', 'infection', 'microbiome']):
            return 'Immune System'
        elif any(word in title_lower for word in ['plant', 'arabidopsis', 'root']):
            return 'Plant Biology'
        elif any(word in title_lower for word in ['radiation', 'dna', 'genomic']):
            return 'Radiation Effects'
        elif any(word in title_lower for word in ['microgravity', 'gravity']):
            return 'Microgravity Adaptation'
        else:
            return 'Other'
    
    def detect_organism(title):
        title_lower = title.lower()
        if 'mouse' in title_lower or 'mice' in title_lower:
            return 'Mouse'
        elif 'arabidopsis' in title_lower:
            return 'Arabidopsis'
        elif 'drosophila' in title_lower:
            return 'Drosophila'
        elif 'human' in title_lower or 'astronaut' in title_lower:
            return 'Human'
        elif 'rat' in title_lower:
            return 'Rat'
        else:
            return 'Various'
    
    df['topic'] = df['Title'].apply(categorize_topic)
    df['organism'] = df['Title'].apply(detect_organism)
    
    return df

# Function to summarize paper from URL
def summarize_paper(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded) if downloaded else None
        
        if text:
            # Single summarization pass
            summary = summarize_entire_text(text)
            return summary
        else:
            return None
    except Exception as e:
        st.error(f"Error summarizing paper: {str(e)}")
        return None

# Simple summarizer function
def summarize_from_url(url):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded) if downloaded else None

    if text:
        # Single summarization pass
        summary = summarize_entire_text(text)
        return summary
    else:
        return "❌ Failed to extract text from the URL."

# Filter publications
def filter_publications(df, search_term, selected_topics, selected_organisms):
    filtered_df = df.copy()
    
    # Only apply topic filter if topics are selected
    if selected_topics and len(selected_topics) > 0:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topics)]
    
    # Only apply organism filter if organisms are selected
    if selected_organisms and len(selected_organisms) > 0:
        filtered_df = filtered_df[filtered_df['organism'].isin(selected_organisms)]
    
    # Only apply search filter if search term is provided
    if search_term and search_term.strip():
        search_terms = search_term.lower().split()
        search_mask = pd.Series(True, index=filtered_df.index)
        for term in search_terms:
            term_mask = (
                filtered_df['Title'].str.lower().str.contains(term, na=False) |
                filtered_df['topic'].str.lower().str.contains(term, na=False) |
                filtered_df['organism'].str.lower().str.contains(term, na=False)
            )
            search_mask &= term_mask
        filtered_df = filtered_df[search_mask]
    
    return filtered_df

# Get research insights
def get_research_insights(df):
    top_topics = df['topic'].value_counts().head(3)
    topics_text = "### 🎯 Most Studied Areas\n"
    for topic, count in top_topics.items():
        topics_text += f"- **{topic}**: {count} publications\n"
    
    gaps_text = "### 🔍 Research Gaps\n"
    gaps = [
        "Limited long-duration human studies",
        "Combined radiation + microgravity effects",
        "Psychological health in space"
    ]
    for gap in gaps:
        gaps_text += f"- {gap}\n"
    
    return topics_text + "\n" + gaps_text

# Main app
def main():
    st.title("🚀 NASA Bioscience Explorer")
    st.markdown("Explore 608 NASA life sciences publications")
    
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("Failed to load data. Please check if the data file exists.")
        return
    
    # Initialize session state for caching summaries
    if 'summary_cache' not in st.session_state:
        st.session_state.summary_cache = {}
    
    # Create header section with filters
    st.markdown("### 🔍 Search and Filter Publications")
    
    # Create three columns for filters
    search_col, topic_col, organism_col = st.columns([1, 1, 1])
    
    with search_col:
        search_term = st.text_input(
            "Search publications:", 
            placeholder="Enter keywords..."
        )
    
    with topic_col:
        topic_options = df['topic'].unique().tolist()
        selected_topics = st.multiselect(
            "Research Topics:",
            options=topic_options,
            default=[]  # Empty by default for better UX
        )
    
    with organism_col:
        organism_options = df['organism'].unique().tolist()
        selected_organisms = st.multiselect(
            "Organisms:",
            options=organism_options,
            default=[]  # Empty by default for better UX
        )
    
    # Add a visual separator
    st.markdown("---")
    
    # Filter data
    filtered_df = filter_publications(df, search_term, selected_topics, selected_organisms)
    
    # Metrics with improved styling
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Publications", len(df))
    col2.metric("Filtered Publications", len(filtered_df))
    col3.metric("Research Topics", df['topic'].nunique())
    col4.metric("Organisms Studied", df['organism'].nunique())
    
    # Create tabs with emoji icons
    tab1, tab2 = st.tabs(["📊 Research Dashboard", "📄 Paper Summarizer"])
    
    with tab1:
        # Visualizations
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Topic distribution pie chart
                topic_counts = filtered_df['topic'].value_counts()
                topic_labels = [f"{topic} ({count})" for topic, count in topic_counts.items()]
                fig1 = px.pie(
                    values=topic_counts.values,
                    names=topic_labels,
                    title="📈 Research Topics Distribution"
                )
                fig1.update_traces(textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Organism distribution bar chart
                organism_counts = filtered_df['organism'].value_counts().reset_index()
                organism_counts.columns = ['Organism', 'Count']
                organism_counts['Label'] = organism_counts.apply(lambda x: f"{x['Organism']} ({x['Count']})", axis=1)
                fig2 = px.bar(
                    data_frame=organism_counts,
                    x='Label',
                    y='Count',
                    title="🧬 Publications by Organism"
                )
                fig2.update_xaxes(tickangle=45)
                fig2.update_layout(xaxis_title="")
                st.plotly_chart(fig2, use_container_width=True)
        
        # Visual separator
        st.markdown("---")
        
        # Publication browser
        st.subheader("📚 Publication Browser")
        
        if not filtered_df.empty:
            for idx, row in filtered_df.iterrows():
                with st.expander(f"**{row['Title']}**"):
                    st.write(f"**Topic:** {row['topic']}")
                    st.write(f"**Organism:** {row['organism']}")
                    st.markdown(f"[📄 Read Paper]({row['Link']})")
                    
                    # Create a unique key for each summary button
                    summary_key = f"summary_{idx}"
                    
                    # Check if summary has been generated for this paper
                    if summary_key not in st.session_state:
                        st.session_state[summary_key] = None
                    
                    # Add summarize button for each publication
                    if st.button("📝 Generate Summary", key=f"btn_{idx}"):
                        with st.spinner("Generating summary..."):
                            summary = summarize_paper(row['Link'])
                            if summary:
                                st.session_state[summary_key] = summary
                            else:
                                st.error("❌ Failed to extract text from this paper.")
                    
                    # Display summary if it exists
                    if st.session_state[summary_key]:
                        st.subheader("Summary")
                        st.write(st.session_state[summary_key])
        else:
            st.warning("🔍 No publications match the current filters.")
        
        # Research insights section
        st.markdown("---")
        st.subheader("💡 Research Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Most Studied Areas")
            top_topics = df['topic'].value_counts().head(3)
            for topic, count in top_topics.items():
                st.write(f"- **{topic}**: {count} publications")
        
        with col2:
            st.markdown("### 🔍 Research Gaps")
            gaps = [
                "Limited long-duration human studies",
                "Combined radiation + microgravity effects",
                "Psychological health in space"
            ]
            for gap in gaps:
                st.write(f"- {gap}")
    
    with tab2:
        st.markdown("### 📄 Research Paper Summarizer")
        st.markdown("Enter any scientific article URL to get an AI-generated summary")
        
        # URL input with better styling
        url_input = st.text_input(
            "Enter Article URL:",
            value="https://pmc.ncbi.nlm.nih.gov/articles/PMC10772081/",
            placeholder="https://pmc.ncbi.nlm.nih.gov/articles/...",
            key="url_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            # Summarize button with primary styling
            if st.button("🚀 Summarize Paper", type="primary"):
                if url_input:
                    if url_input not in st.session_state.summary_cache:
                        with st.spinner("📝 Generating summary..."):
                            summary = summarize_from_url(url_input)
                            st.session_state.summary_cache[url_input] = summary

        with col2:
            if url_input in st.session_state.summary_cache:
                st.info(st.session_state.summary_cache[url_input])
            elif not url_input:
                st.warning("⚠️ Please enter a URL to summarize")

if __name__ == "__main__":
    main()
