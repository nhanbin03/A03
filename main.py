from calendar import c
import os
import json
import pickle
from underthesea import word_tokenize
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Step 1: Preprocessing Functions
def preprocess_text(text):
    # Convert to lowercase and normalize diacritics
    text = unidecode(text.lower())
    # Tokenize using Underthesea
    tokens = word_tokenize(text, format="text")
    # Remove Vietnamese stop words
    vietnamese_stopwords = set(["và", "là", "của", "các", "để", "với", "nhưng"])
    tokens = [word for word in tokens.split() if word not in vietnamese_stopwords]
    return " ".join(tokens)

# Step 2: Load and Preprocess Documents
def load_json_data(folder_path):
    documents = []
    titles = []
    contents = []
    dates = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    title = data.get("title", "No Title")
                    content = data.get("content", "No Content")
                    date = data.get("date", "Unknown Date")
                    full_text = f"{title} {content}"  # Combine title and content
                    processed_text = preprocess_text(full_text)
                    documents.append(processed_text)
                    titles.append(title)
                    contents.append(content)
                    dates.append(date)
                except json.JSONDecodeError:
                    st.error(f"Error reading JSON file: {file_name}")
    return documents, titles, contents, dates


# Step 3: Create TF-IDF Index
def create_index(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    # Save the index and vectorizer for later use
    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
    print("Index created and saved successfully.")

# Step 4: Load the Index
def load_index(documents):
    # Check if the index file exists
    if not os.path.exists("tfidf_matrix.pkl"):
        create_index(documents)
    
    with open("tfidf_matrix.pkl", "rb") as f:
        vectorizer, tfidf_matrix = pickle.load(f)
    return vectorizer, tfidf_matrix

# Step 5: Search Function
def search(query, vectorizer, tfidf_matrix, titles, contents, dates):
    query = preprocess_text(query)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Filter results with scores greater than 0
    results = []
    for idx, score in enumerate(similarities):
        if score > 0:
            snippet = contents[idx][:150] + "..." if len(contents[idx]) > 150 else contents[idx]
            snippet = snippet.replace("\n", " ")  # Remove newlines from snippet
            results.append({
                "title": titles[idx],
                "snippet": snippet,
                "date": dates[idx],
                "score": score,
                "content": contents[idx],
                "key": idx
            })

    return results


# Streamlit Interface
def main():
    st.title("Vietnamese Text Search Engine")

    # Step 1: Load data and create index only once when the app starts
    if 'vectorizer' not in st.session_state:
        with st.spinner("Loading data and creating index..."):
            folder_path = "./data"
            if os.path.exists(folder_path):
                documents, titles, contents, dates = load_json_data(folder_path)
                if documents:
                    vectorizer, tfidf_matrix = load_index(documents)
                    st.session_state.vectorizer = vectorizer
                    st.session_state.tfidf_matrix = tfidf_matrix
                    st.session_state.documents = documents
                    st.session_state.titles = titles
                    st.session_state.contents = contents
                    st.session_state.dates = dates
                else:
                    st.error("No valid JSON files found in the data folder.")
                    return
            else:
                st.error("Data folder './data/' not found.")
                return
        st.success("Data loaded successfully!")

    # Step 2: Search functionality
    st.header("Search")
    query = st.text_input("Enter your search query:")

    # Dropdown to select sorting method
    sort_option = st.selectbox(
        "Sort by:",
        options=["Sort by: Relevance", "Sort by: Newest"],
        label_visibility="collapsed",
    )

    if query:
        with st.spinner("Searching..."):
            results = search(
                query,
                st.session_state.vectorizer,
                st.session_state.tfidf_matrix,
                st.session_state.titles,
                st.session_state.contents,
                st.session_state.dates,
            )

            if results:
                # Apply sorting based on user's choice
                if sort_option == "Sort by: Relevance":
                    # Sort results by date (assuming date format is DD/MM/YYYY HH:MM GMT+7)
                    results.sort(key=lambda x: x["date"], reverse=True)
                elif sort_option == "Sort by: Newest":
                    # Sort by score (default sorting in the `search` function)
                    results.sort(key=lambda x: x["score"], reverse=True)

                st.subheader("Search Results:")
                for rank, result in enumerate(results, start=1):
                    # Show the title, date, score, and snippet
                    st.markdown(f"""
                        ### Rank {rank}: {result['title']}
                        **Date**: {result['date']}  
                        **Score**: {result['score']:.4f}  
                        **Snippet**: {result['snippet']}
                    """)

                    # Add a button to show the full content
                    content_key = f"content_{result['key']}"
                    if st.button("Toggle Full Content", key=f"toggle_{result['key']}"):
                        st.session_state[content_key] = not st.session_state.get(content_key, False)

                    if st.session_state.get(content_key, False):
                        st.markdown(result['content'].replace("\n", "<br>"), unsafe_allow_html=True)
            else:
                st.warning("No results found.")

# Run the Streamlit app
if __name__ == "__main__":
    main()