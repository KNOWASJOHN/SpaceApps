import streamlit as st
import pandas as pd
import plotly.express as px
import trafilatura
from transformers import pipeline
import textwrap

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="NASA Bioscience Explorer",
    page_icon="https://github.com/KNOWASJOHN/SpaceApps/blob/main/kryonix.jpg?raw=true",
    layout="wide",
    initial_sidebar_state="collapsed"  # Hide the sidebar by default
)

# Load the summarizer
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

# Summarization function for individual sections
def summarize_section(text, max_length=100, min_length=40):
    try:
        summarizer = load_summarizer()
        
        # Check if text is too short
        if not text or len(text.strip()) < 50:
            return "Insufficient text for summary."
        
        # Limit input to 2000 characters for sections
        text = text[:4000] if len(text) > 4000 else text
        
        result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Error summarizing section: {str(e)}"

# SECTION EXTRACTION FUNCTION with improved logic
def extract_sections(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
            
        full_text = trafilatura.extract(downloaded)
        if not full_text:
            return None
        
        sections = {}
        lines = full_text.split('\n')
        current_section = None
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
                
            # Check for section headers (more precise matching)
            line_lower = line_clean.lower()
            
            # Reset current section if we find a new section header
            if len(line_clean) < 100:  # Likely a header if short
                if 'introduction' in line_lower and current_section != 'introduction':
                    current_section = 'introduction'
                    sections[current_section] = []
                    continue
                elif 'results' in line_lower and current_section != 'results':
                    current_section = 'results'
                    sections[current_section] = []
                    continue
                elif 'conclusion' in line_lower and current_section != 'conclusion':
                    current_section = 'conclusion'
                    sections[current_section] = []
                    continue
                elif ('methods' in line_lower or 'methodology' in line_lower) and current_section != 'methods':
                    current_section = 'methods'
                    sections[current_section] = []
                    continue
                elif 'discussion' in line_lower and current_section != 'discussion':
                    current_section = 'discussion'
                    sections[current_section] = []
                    continue
                elif 'abstract' in line_lower and current_section != 'abstract':
                    current_section = 'abstract'
                    sections[current_section] = []
                    continue
            
            # Add content to current section if we're in one
            if current_section and current_section in sections:
                sections[current_section].append(line_clean)
        
        # Convert lists to strings and limit each section
        processed_sections = {}
        for section, content in sections.items():
            if content:
                section_text = ' '.join(content)
                # Limit to 2000 characters per section
                section_text = section_text[:2000]
                if len(section_text) > 50:  # Only include if meaningful content
                    processed_sections[section] = section_text
        
        return processed_sections if processed_sections else None
        
    except Exception as e:
        st.error(f"Error extracting sections: {str(e)}")
        return None

# FALLBACK FUNCTION: Use full text if section extraction fails
def extract_full_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded) if downloaded else None
        return text[:4000] if text else None
    except Exception as e:
        return None

# UPDATED Function to summarize paper from URL with separate section summaries
def summarize_paper(url):
    try:
        # Extract specific sections
        sections = extract_sections(url)
        
        if sections:
            # Summarize each section individually
            section_summaries = {}
            for section_name, section_text in sections.items():
                if section_text and len(section_text) > 100:
                    section_summary = summarize_section(section_text)
                    section_summaries[section_name] = section_summary
            
            return section_summaries if section_summaries else None
        else:
            # Fallback to full text extraction
            st.warning("⚠️ Could not extract specific sections, using full text instead.")
            full_text = extract_full_text(url)
            if full_text and len(full_text) > 100:
                # For full text, create a single summary but label it as "Overall Summary"
                overall_summary = summarize_section(full_text, max_length=150, min_length=60)
                return {"Overall Summary": overall_summary}
            else:
                return None
                
    except Exception as e:
        st.error(f"Error summarizing paper: {str(e)}")
        return None

# UPDATED Simple summarizer function with separate sections
def summarize_from_url(url):
    try:
        # Extract sections and summarize each
        sections = extract_sections(url)
        
        if sections:
            section_summaries = {}
            for section_name, section_text in sections.items():
                if section_text and len(section_text) > 100:
                    section_summary = summarize_section(section_text)
                    section_summaries[section_name] = section_summary
            
            return section_summaries if section_summaries else None
        else:
            # Fallback to full text
            full_text = extract_full_text(url)
            if full_text and len(full_text) > 100:
                overall_summary = summarize_section(full_text, max_length=150, min_length=60)
                return {"Overall Summary": overall_summary}
            else:
                return "❌ Failed to extract meaningful text from the URL."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Load data
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

# Function to display section summaries in a nice format
def display_section_summaries(summaries, use_expander=False):
    if not summaries:
        return
    
    # Display each section summary either in an expander or container
    for section_name, summary_text in summaries.items():
        if use_expander:
            with st.expander(f"📋 {section_name.title()} Summary"):
                st.info(summary_text)
        else:
            st.markdown(f"**📋 {section_name.title()} Summary**")
            st.info(summary_text)
            st.markdown("---")

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
            default=[]
        )
    
    with organism_col:
        organism_options = df['organism'].unique().tolist()
        selected_organisms = st.multiselect(
            "Organisms:",
            options=organism_options,
            default=[]
        )
    
    st.markdown("---")
    
    # Filter data
    filtered_df = filter_publications(df, search_term, selected_topics, selected_organisms)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Publications", len(df))
    col2.metric("Filtered Publications", len(filtered_df))
    col3.metric("Research Topics", df['topic'].nunique())
    col4.metric("Organisms Studied", df['organism'].nunique())
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Research Dashboard", "📄 Paper Summarizer"])
    
    with tab1:
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
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
        
        st.markdown("---")
        st.subheader("📚 Publication Browser")
        
        if not filtered_df.empty:
            for idx, row in filtered_df.iterrows():
                # Create an expander for each paper
                with st.expander(f"📑 {row['Title']}", expanded=False):
                    st.write(f"**Topic:** {row['topic']}")
                    st.write(f"**Organism:** {row['organism']}")
                    st.markdown(f"[📄 Read Paper]({row['Link']})")
                    
                    summary_key = f"summary_{idx}"
                    
                    if summary_key not in st.session_state:
                        st.session_state[summary_key] = None
                    
                    if st.button("📝 Generate Summary", key=f"btn_{idx}"):
                        with st.spinner("Generating section summaries..."):
                            summaries = summarize_paper(row['Link'])
                            if summaries:
                                st.session_state[summary_key] = summaries
                            else:
                                st.error("❌ Failed to extract text from this paper.")
                    
                    if st.session_state[summary_key]:
                        st.write("📋 Section Summaries:")
                        display_section_summaries(st.session_state[summary_key], use_expander=False)
        else:
            st.warning("🔍 No publications match the current filters.")
        
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
        st.markdown("Enter any scientific article URL to get AI-generated section summaries")
        
        url_input = st.text_input(
            "Enter Article URL:",
            value="https://pmc.ncbi.nlm.nih.gov/articles/PMC10772081/",
            placeholder="https://pmc.ncbi.nlm.nih.gov/articles/...",
            key="url_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Summarize Paper", type="primary"):
                if url_input:
                    if url_input not in st.session_state.summary_cache:
                        with st.spinner("📝 Generating section summaries..."):
                            summaries = summarize_from_url(url_input)
                            st.session_state.summary_cache[url_input] = summaries

        with col2:
            if url_input in st.session_state.summary_cache:
                summaries = st.session_state.summary_cache[url_input]
                if isinstance(summaries, dict):
                    display_section_summaries(summaries, use_expander=True)  # Use expanders in the Paper Summarizer tab
                else:
                    st.info(summaries)
            elif not url_input:
                st.warning("⚠️ Please enter a URL to summarize")

if __name__ == "__main__":
    main()