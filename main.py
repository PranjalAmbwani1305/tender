import streamlit as st
import pinecone
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Load Hugging Face models
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 768D embedding model
generator = pipeline("text-generation", model="gpt2", max_length=250)  # GPT-2 for tender text

# Initialize Pinecone
pinecone.init(api_key=st.secrets["PINECONE_API_KEY"], environment="us-west1-gcp")
index_name = "tender-docs"

# Ensure Pinecone index exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=768, metric="cosine")

index = pinecone.Index(index_name)

# Tender document structure
TENDER_SECTIONS = [
    "Invitation to Bid",
    "Project Overview",
    "Bidder Instructions",
    "Scope of Work",
    "Contract Terms",
    "Financial Proposal"
]

def get_embedding(text):
    """Generate a 768D embedding for given text."""
    return embedder.encode(text).tolist()

def find_similar_sections(query, top_k=3):
    """Search for similar tender sections in Pinecone."""
    query_vector = get_embedding(query)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results.get("matches", [])

def generate_tender_section(section, details):
    """Generate content for a specific tender section."""
    prompt = f"{section}\nProject: {details}\n\nContent:"
    generated_text = generator(prompt, num_return_sequences=1)[0]["generated_text"]
    return generated_text

def main():
    st.set_page_config(page_title="AI Tender Generator", layout="wide")
    st.title("AI Tender Generator")
    st.markdown("Generate professional tender documents effortlessly.")

    with st.form("tender_form"):
        title = st.text_input("Project Title")
        location = st.text_input("Project Location")
        duration = st.number_input("Project Duration (months)", min_value=1, value=6)
        budget = st.text_input("Project Budget")
        description = st.text_area("Project Description", help="Provide relevant project details")
        generate_btn = st.form_submit_button("Generate Tender")

    if generate_btn:
        if not title or not location or not description:
            st.error("Please fill in all required fields.")
            return

        project_details = f"Title: {title}, Location: {location}, Duration: {duration} months, Budget: {budget or 'Not specified'}, Description: {description}"

        with st.spinner("Generating tender document..."):
            tender_content = {}
            for section in TENDER_SECTIONS:
                tender_content[section] = generate_tender_section(section, project_details)

            st.success("Tender document generated successfully!")

            for section, content in tender_content.items():
                st.subheader(section)
                st.write(content)

if __name__ == "__main__":
    main()
