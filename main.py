import streamlit as st
from fpdf import FPDF
from io import BytesIO
import pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import os


pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')


index_name = 'tender-index'
try:
    index = pinecone.Index(index_name)
except Exception as e:
    index = pinecone.create_index(index_name, dimension=768, metric='cosine')


model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_pdf(new_tender):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="PROJECT DETAILS", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Title: {new_tender['title']}", ln=True)
    pdf.cell(200, 10, txt=f"Budget: {new_tender['budget']}", ln=True)
    pdf.cell(200, 10, txt=f"Duration: {new_tender['duration']}", ln=True)
    pdf.cell(200, 10, txt=f"Location: {new_tender['location']}", ln=True)
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Description: {new_tender['description']}")
    
    # Page 2
    
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="NOTICE INVITING TENDER", ln=True, align="C")
    
    # Dynamic content from project
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"e-TENDER NO: {new_tender['tender_number']}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Name of Work: {new_tender['title']}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Location of work: {new_tender['location']}")
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Interested bidders may please download the Tender  from our website: {new_tender['website']}")
    

    
    
    pdf.ln(10)
    pdf.set_font("Arial", size=10, style='I')
    pdf.multi_cell(0, 10, "Tender Bid Submission Instructions and Terms.\n"
                           "For detailed instructions, please visit our website: www.ourwebsite.com")
    
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    return pdf_buffer


def add_to_pinecone(tender_texts, tender_ids):
    embeddings = model.encode(tender_texts)
    index.upsert(vectors=zip(tender_ids, embeddings))


def search_pinecone(query, top_k=3):
    query_embedding = model.encode([query])
    result = index.query(query_embedding, top_k=top_k, include_values=True)
    return result['matches']


st.title("Tender Generator Bot")


title = st.text_input("Tender Title")
budget = st.number_input("Budget", min_value=0)
duration = st.text_input("Duration (in months)")
location = st.text_input("Location")
description = st.text_area("Description")
introduction = st.text_area("Introduction")
scope_of_work = st.text_area("Scope of Work")
bidding_details = st.text_area("Bidding Details")
old_sample = st.text_area("Old Tender Sample")


tender_number = st.text_input("e-Tender Number")
website = st.text_input("Tender Website URL")


new_tender = {
    'title': title,
    'budget': f'₹{budget}',
    'duration': f'{duration} Months',
    'location': location,
    'description': description,
    'introduction': introduction,
    'scope_of_work': scope_of_work,
    'bidding_details': bidding_details,
    'old_sample': old_sample,
    'tender_number': tender_number,
    'website': website
}


if st.button("Generate Tender"):
    add_to_pinecone([old_sample], ["sample_id_1"])  
    
 
    similar_tenders = search_pinecone(new_tender['description'])
    

    pdf_buffer = generate_pdf(new_tender)
    
    st.subheader("Similar Old Tenders Found:")
    for match in similar_tenders:
        st.write(f"Tender ID: {match['id']} - Score: {match['score']}")
        st.text(match['metadata']['text'])
    
    st.download_button(
        label="Download Tender as PDF",
        data=pdf_buffer,
        file_name="new_tender.pdf",
        mime="application/pdf"
    )
