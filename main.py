import os
import shutil
import torch
import streamlit as st
from pathlib import Path
import pinecone
from transformers import AutoTokenizer, AutoModel
from PyPDF2 import PdfReader
import re

# Initialize Streamlit secrets
api_key = st.secrets["pinecone"]["api_key"]
env = st.secrets["pinecone"]["ENV"]
index_name = st.secrets["pinecone"]["INDEX_NAME"]
hf_token = st.secrets["huggingface"]["token"]

# Initialize Pinecone
pinecone.init(api_key=api_key, environment=env)
index = pinecone.Index(index_name)

# Load Transformer Model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModel.from_pretrained(model_name, token=hf_token)

# Define Storage Folder
storage_folder = "content_storage"
os.makedirs(storage_folder, exist_ok=True)

def extract_text_sections(text):
    """Extract structured sections dynamically from the PDF text."""
    sections = []
    lines = text.split("\n")
    current_section = []

    for line in lines:
        line = line.strip()

        # Identify section headings
        if re.match(r"^[A-Z ]{5,}$", line) or line.endswith(":"):
            if current_section:
                sections.append("\n".join(current_section))  # Store previous section
            current_section = [f"**{line}**"]  # Start new section
        else:
            current_section.append(line)

    if current_section:
        sections.append("\n".join(current_section))  # Store last section

    return sections

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file and formats it into structured sections."""
    reader = PdfReader(file_path)
    extracted_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)

    return extract_text_sections("\n".join(extracted_text)) if extracted_text else []

def embed_text(text):
    """Generate an embedding for the given text chunk."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().tolist()
    
    return embedding

def store_in_pinecone(file_name, text_sections):
    """Store extracted content in Pinecone with embeddings."""
    for i, section in enumerate(text_sections):
        chunk_id = f"{file_name}_chunk_{i}"
        embedding = embed_text(section)
        metadata = {"file_name": file_name, "chunk_id": i, "text": section}

        expected_dim = len(embed_text("test"))  # Dynamically get model output size
        if len(embedding) == expected_dim:
            index.upsert([(chunk_id, embedding, metadata)])
            st.write(f"✅ Stored {file_name} - section {i} in Pinecone.")
        else:
            st.error(f"❌ Invalid vector size: Expected {expected_dim}, got {len(embedding)}.")

def process_folder(folder_path):
    """Process all PDFs in a folder and store structured content in Pinecone."""
    st.write(f"📂 Processing folder: {folder_path}")

    for file in Path(folder_path).rglob("*.pdf"):
        st.write(f"📄 Processing {file.name}...")
        structured_content = extract_text_from_pdf(file)

        if structured_content:
            store_in_pinecone(file.name, structured_content)
        else:
            st.warning(f"⚠️ No structured content found in {file.name}.")

def main():
    st.title("📌 Process & Store Structured PDF Data in Pinecone")

    uploaded_zip = st.file_uploader("📁 Upload a ZIP folder containing PDFs", type=["zip"])

    if uploaded_zip:
        zip_path = os.path.join(storage_folder, uploaded_zip.name)
        
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getvalue())

        folder_extract_path = os.path.join(storage_folder, "extracted_files")
        shutil.unpack_archive(zip_path, folder_extract_path)
        st.write(f"📂 Folder extracted to {folder_extract_path}")

        process_folder(folder_extract_path)
        st.success("✅ All structured data stored in Pinecone!")

if __name__ == "__main__":
    main()
