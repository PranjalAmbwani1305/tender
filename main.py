import os
from transformers import BertTokenizer, BertModel
import torch
import pinecone

# Initialize Pinecone
pinecone.init(api_key="your-pinecone-api-key", environment="us-west1-gcp")


index = pinecone.Index("tender-docs")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Average pooling of token embeddings


def insert_old_tenders_to_pinecone(folder_path):
    tender_files = os.listdir(folder_path)
    for file in tender_files:
        if file.endswith(".txt"):  # Assuming old tenders are in text files
            with open(os.path.join(folder_path, file), 'r') as f:
                tender_text = f.read()
                embedding = generate_embeddings(tender_text)
                tender_id = file  # Use file name as unique ID
                index.upsert([(tender_id, embedding)])


insert_old_tenders_to_pinecone('path/to/old_tenders_folder')
