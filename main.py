import streamlit as st
import pinecone
from transformers import pipeline

pinecone.init(api_key="your_pinecone_api_key", environment="us-west1-gcp")
index_name = "tender-bot"
index = pinecone.Index(index_name)

generator = pipeline("text-generation", model="gpt2")

st.title('AI Tender Generator')
st.write('Fill out the details below to generate your tender document.')

project_title = st.text_input('Project Title')
project_location = st.text_input('Project Location')
project_duration = st.number_input('Project Duration (months)', min_value=1)
project_budget = st.number_input('Project Budget')
project_description = st.text_area('Project Description')

if st.button('Generate Tender'):
    query = f"Project Title: {project_title}, Location: {project_location}, Description: {project_description}"
    results = index.query(query, top_k=1)

    tender_content = generator(f"Create a professional tender for {project_title} located at {project_location} with description {project_description}. Budget: {project_budget}.")

    tender_text = tender_content[0]['generated_text']

    st.subheader("Generated Tender Document:")
    st.text_area('Tender Document', value=tender_text, height=300)

    st.download_button('Download Tender Document', tender_text, file_name="tender_document.txt")
