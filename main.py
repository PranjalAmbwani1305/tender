import os
import streamlit as st
import pinecone
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API keys from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
hf_api_key = os.getenv("HF_API_KEY")

# Initialize Pinecone with the API key
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")
index_name = "tender-bot"
index = pinecone.Index(index_name)

# Login to Hugging Face using the API key
login(token=hf_api_key)

# Initialize Hugging Face pipeline for text generation
generator = pipeline("text-generation", model="gpt2")

# Streamlit UI
st.title('AI Tender Generator')
st.write('Fill out the details below to generate your tender document.')

# Input fields for the project details
project_title = st.text_input('Project Title', required=True)
project_location = st.text_input('Project Location', required=True)
project_duration = st.number_input('Project Duration (months)', min_value=1, required=True)
project_budget = st.number_input('Project Budget')
project_description = st.text_area('Project Description', required=True)

if st.button('Generate Tender'):
    # Query Pinecone for relevant data
    query = f"Project Title: {project_title}, Location: {project_location}, Description: {project_description}"
    results = index.query(query, top_k=1)

    # Generate Tender Content using GPT-2 model
    tender_content = generator(f"Create a professional tender for {project_title} located at {project_location} with description {project_description}. Budget: {project_budget}.")

    tender_text = tender_content[0]['generated_text']

    # Creating the tender document structure
    tender_document = f"""
    Tender document generated successfully!

    NOTICE INVITING TENDER

    E-TENDER NO. 003/ERP/UPGRADE/2021-22

    We invite e-tenders for the following Work:
    - Name of Work: {project_title}
    - Location of Work: {project_location}
    - Budget: {project_budget}

    BRIEF INTRODUCTION:
    {tender_text}

    INSTRUCTION TO BIDDERS:
    - Please read the instructions carefully before bidding.
    - Ensure all required documents are attached.

    SCOPE OF WORK:
    {project_description}

    TERMS AND CONDITIONS:
    - All bids must be submitted online via the official website.
    - Late submissions will not be considered.

    PRICE BID:
    - Submit your price bids with the required documents.

    Interested bidders may please download the Tender Document from our website: www.ourwebsite.com

    I/C. General Manager,
    """

    # Display generated tender document
    st.subheader("Generated Tender Document:")
    st.text_area('Tender Document', value=tender_document, height=300)

    # Download button
    st.download_button('Download Tender Document', tender_document, file_name="tender_document.txt")
