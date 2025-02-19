import streamlit as st
import pinecone
from transformers import pipeline
from huggingface_hub import login

# Load secrets from Streamlit
pinecone_api_key = st.secrets["PINECONE"]["API_KEY"]
pinecone_env = st.secrets["PINECONE"]["ENVIRONMENT"]
hf_api_key = st.secrets["HUGGINGFACE"]["API_KEY"]

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index_name = "tender-bot"
index = pinecone.Index(index_name)

# Login to Hugging Face
login(token=hf_api_key)

# Initialize text generation model
generator = pipeline("text-generation", model="gpt2")

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
        # Query Pinecone (Optional - if you have embeddings stored)
        query = f"Project Title: {project_title}, Location: {project_location}, Description: {project_description}"
        results = index.query(query, top_k=1)

        # Generate tender document
        tender_content = generator(
            f"Create a professional tender for {project_title} located at {project_location}. "
            f"Description: {project_description}. Budget: {project_budget}.",
            max_length=300
        )

        tender_text = tender_content[0]['generated_text']

        # Format tender document
        tender_document = f"""
        Tender document generated successfully!

        **NOTICE INVITING TENDER**

        **E-TENDER NO. 003/ERP/UPGRADE/2021-22**

        **Work Details:**
        - **Project Title:** {project_title}
        - **Location:** {project_location}
        - **Duration:** {project_duration} months
        - **Budget:** {project_budget}

        **Project Description:**
        {project_description}

        **Tender Details:**
        {tender_text}

        Interested bidders may please download the Tender Document from our website: www.ourwebsite.com

        I/C. General Manager
        """

        # Display generated tender
        st.subheader("Generated Tender Document")
        st.text_area("Tender Document", value=tender_document, height=300)

        # Provide download button
        st.download_button("Download Tender Document", tender_document, file_name="tender_document.txt")
