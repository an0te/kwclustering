import streamlit as st
import pandas as pd
import openai
import io
import plotly.graph_objects as go
from collections import Counter
import logging
import time

# Configure logging to write to terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(page_title="Keyword Classifier", layout="wide")

# Sidebar
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; padding: 10px;">
        <img src="https://i.imgur.com/CU9o9Ta.gif" width="25">
        <span style="margin-left: 10px; font-weight: bold;">Developed with ❤️ by Andreas</span>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("About")
st.sidebar.write("""
This tool allows you to classify and analyze keywords using OpenAI's GPT model. You can:
- Upload a CSV file with keywords (and optionally, their volumes) or enter keywords manually
- Specify categories for classification, including descriptions and example keywords
- Visualize the distribution of keywords across categories
- Download the results as a CSV file
""")

# OpenAI API Key input
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    openai.api_key = api_key
else:
    st.sidebar.warning("Please enter your OpenAI API Key to use this tool.")

st.title("🚀 Bulk Keyword Classifier & Analyzer 📊")
st.write("Unlock the power of AI to cluster and analyze your keywords like never before! 🔍💡")

def classify_keywords(keywords, categories):
    results = []
    progress_bar = st.progress(0)
    for i, keyword in enumerate(keywords):
        try:
            category_info = "\n".join([f"- {cat['name']}: {cat['description']} (Examples: {', '.join(cat['examples'])})" for cat in categories])
            messages = [
                {"role": "system", "content": "You are a keyword classifier. Respond only with the category name."},
                {"role": "user", "content": f"Classify the following keyword into one of these categories:\n{category_info}\n\nKeyword: {keyword}"}
            ]
            logging.info(f"Classifying keyword: {keyword}")
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.5,
                timeout=10  # Set a 10-second timeout
            )
            category = response.choices[0].message['content'].strip()
            results.append((keyword, category))
            logging.info(f"Classified {keyword} as {category}")
        except Exception as e:
            logging.error(f"Error classifying keyword {keyword}: {str(e)}")
            results.append((keyword, "Error"))
        progress_bar.progress((i + 1) / len(keywords))
        time.sleep(0.1)  # Add a small delay to avoid rate limiting
    return results

def main():
    # File upload or text input
    upload_option = st.radio("Choose input method:", ["Upload CSV", "Enter keywords manually"])
    
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.write(df.head())
            keyword_column = st.selectbox("Select the column containing keywords:", df.columns)
            volume_column = st.selectbox("Select the column containing search volume (optional):", ["None"] + list(df.columns))
            
            if volume_column == "None":
                keywords = df[keyword_column].tolist()
                volumes = [1] * len(keywords)
            else:
                keywords = df[keyword_column].tolist()
                volumes = df[volume_column].tolist()
    else:
        keywords_input = st.text_area("Enter keywords (one per line):")
        keywords = [kw.strip() for kw in keywords_input.split("\n") if kw.strip()]
        volumes = [1] * len(keywords)
    
    # Category input
    st.subheader("📋 Category Input")
    st.write("Enter categories, their descriptions, and example keywords below:")
    
    categories = []
    category_count = st.number_input("Number of categories", min_value=1, value=3, step=1)
    
    for i in range(category_count):
        col1, col2, col3 = st.columns(3)
        with col1:
            name = st.text_input(f"Category {i+1} Name", key=f"cat_name_{i}")
        with col2:
            description = st.text_input(f"Category {i+1} Description", key=f"cat_desc_{i}")
        with col3:
            examples = st.text_input(f"Category {i+1} Example Keywords (comma-separated)", key=f"cat_ex_{i}")
        
        if name and description and examples:
            categories.append({
                "name": name,
                "description": description,
                "examples": [ex.strip() for ex in examples.split(",")]
            })
    
    if st.button("🔍 Analyze Keywords"):
        if not keywords or not categories:
            st.warning("Please provide both keywords and categories.")
        elif not api_key:
            st.warning("Please enter your OpenAI API Key in the sidebar to use this tool.")
        else:
            try:
                with st.spinner("Analyzing keywords..."):
                    # Classify keywords
                    st.write("Classifying keywords...")
                    classified_keywords = classify_keywords(keywords, categories)
                    
                    # Prepare data for visualization and results
                    df_results = pd.DataFrame(classified_keywords, columns=["Keyword", "Category"])
                    df_results["Volume"] = volumes
                    
                    # Create bar graph for keyword count
                    category_counts = Counter([category for _, category in classified_keywords])
                    fig_count = go.Figure(data=[go.Bar(x=list(category_counts.keys()), y=list(category_counts.values()))])
                    fig_count.update_layout(title="Keyword Count by Category", xaxis_title="Categories", yaxis_title="Number of Keywords")
                    
                    # Create bar graph for total volume by category
                    category_volumes = df_results.groupby("Category")["Volume"].sum().reset_index()
                    fig_volume = go.Figure(data=[go.Bar(x=category_volumes["Category"], y=category_volumes["Volume"])])
                    fig_volume.update_layout(title="Total Volume by Category", xaxis_title="Categories", yaxis_title="Total Volume")
                    
                    # Display results in three columns
                    st.subheader("🎉 Keyword Classification Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.plotly_chart(fig_count, use_container_width=True)
                    
                    with col2:
                        st.plotly_chart(fig_volume, use_container_width=True)
                    
                    with col3:
                        st.dataframe(df_results, height=400)  # Add scrollbar by setting height
                    
                    # Prepare data for download
                    csv = df_results.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name="keyword_classification_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logging.error(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main()