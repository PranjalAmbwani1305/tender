import streamlit as st
from pinecone import Pinecone
from transformers import pipeline
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer

# Load secrets from Streamlit
pinecone_api_key = st.secrets["PINECONE"]["API_KEY"]
hf_api_key = st.secrets["HUGGINGFACE"]["API_KEY"]

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("tender-bot")

# Login to Hugging Face
HfApi().set_access_token(hf_api_key)

# Initialize text generation model
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct")

# Streamlit UI
st.title("AI Tender Generator")
st.write("Fill out the details below to generate your tender document.")

# Input fields
project_title = st.text_input("Project Title", placeholder="Enter project title")
project_location = st.text_input("Project Location", placeholder="Enter location")
project_duration = st.number_input("Project Duration (months)", min_value=1, step=1)
project_budget = st.number_input("Project Budget", min_value=0, step=1000)
project_description = st.text_area("Project Description", placeholder="Enter details about the project")

# Generate tender button
if st.button("Generate Tender"):
    if not project_title or not project_location or not project_description:
        st.warning("Please fill in all required fields.")
    else:
        # Generate tender document
        tender_content = generator(
            f"Create a professional tender for {project_title} located at {project_location}. "
            f"Description: {project_description}. Budget: {project_budget}.",
            max_length=300
        )

        tender_text = tender_content[0]['generated_text']

        # Format tender document
        tender_document = f"""
        **NOTICE INVITING TENDER**

        **Project Title:** {project_title}  
        **Location:** {project_location}  
        **Duration:** {project_duration} months  
        **Budget:** {project_budget}  

        **Project Description:**  
        {project_description}  

        **Tender Details:**  
        {tender_text}  

        Interested bidders may please download the Tender Document from our website: www.ourwebsite.com  

        **I/C. General Manager**
        """

        # Display generated tender
        st.subheader("Generated Tender Document")
        st.code(tender_document, language="markdown")

        # Provide download button
        st.download_button("Download Tender Document", tender_document, file_name="tender_document.txt")
