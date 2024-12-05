# app.py

import streamlit as st
import pdfplumber
import nltk
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Download NLTK data
nltk.download('punkt', quiet=True)

# Load models
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache(allow_output_mutation=True)
def load_language_model():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    lm_model = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/gpt-neo-2.7B',
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    return tokenizer, lm_model

embedding_model = load_embedding_model()
tokenizer, lm_model = load_language_model()

# PDF text extraction
def extract_text_from_pdf(pdf_file):
    text = ''
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
    return text

# Text preprocessing and chunking
def split_text_into_chunks(text, max_length=500):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) <= max_length:
            chunk += ' ' + sentence
        else:
            chunks.append(chunk.strip())
            chunk = sentence
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Create embeddings and build FAISS index
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, top_k=1):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks

# Generate response using the language model
def generate_response(prompt):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    lm_model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    max_new_tokens = 100  # Adjust as needed
    with torch.no_grad():
        output_ids = lm_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = output_text[len(prompt):].strip()

    # Implement stop tokens
    stop_sequences = ["\nCandidate:", "\nInterviewer:", "Candidate:", "Interviewer:"]
    for stop_sequence in stop_sequences:
        if stop_sequence in generated_text:
            generated_text = generated_text.split(stop_sequence)[0].strip()
            break

    return generated_text

# Create prompt function
def create_prompt(context, conversation_history, user_input):
    prompt = (
        f"You are an interviewer conducting a case interview based on the context provided.\n\n"
        f"### Context:\n{context}\n\n"
        f"### Conversation:\n{conversation_history}"
        f"Candidate: {user_input}\n"
        f"Interviewer:"
    )
    return prompt

# Streamlit interface
def main():
    st.title("McKinsey Case Interview Chatbot")

    # Session state initialization
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []
    if 'index' not in st.session_state:
        st.session_state['index'] = None
    if 'chunks' not in st.session_state:
        st.session_state['chunks'] = None

    # Upload PDF
    pdf_file = st.file_uploader("Upload a Case Study PDF", type=["pdf"])
    if pdf_file and st.session_state.get('index') is None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(pdf_file)
                if text:
                    chunks = split_text_into_chunks(text)
                    index = create_faiss_index(chunks)
                    st.session_state['index'] = index
                    st.session_state['chunks'] = chunks
                    st.success("PDF uploaded and processed.")
                else:
                    st.error("Unable to extract text from the PDF.")

    if st.session_state.get('index') is not None:
        st.markdown("### Chat with the Interviewer")
        user_input = st.text_input("Your response:", "")
        if st.button("Send"):
            if user_input:
                retrieved_chunks = retrieve_relevant_chunks(
                    user_input, st.session_state['index'],
                    st.session_state['chunks'], top_k=1
                )
                context = retrieved_chunks[0] if retrieved_chunks else ''
                # Use only the last few exchanges
                conversation_history = ''
                last_turns = st.session_state['conversation'][-4:]
                for turn in last_turns:
                    conversation_history += f"{turn['role']}: {turn['content']}\n"

                # Define prompt after building conversation history
                prompt = create_prompt(context, conversation_history, user_input)

                with st.spinner("Generating response..."):
                    response = generate_response(prompt)
                st.session_state['conversation'].append({'role': 'Candidate', 'content': user_input})
                st.session_state['conversation'].append({'role': 'Interviewer', 'content': response})
                st.text_area("Interviewer:", value=response, height=200)

        # Display conversation history
        st.markdown("### Conversation History")
        for turn in st.session_state['conversation']:
            st.write(f"**{turn['role']}:** {turn['content']}")

if __name__ == "__main__":
    main()
