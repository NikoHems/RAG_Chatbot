# McKinsey Case Interview Chatbot

A Streamlit application designed to simulate a McKinsey-style case interview using natural language processing and machine learning. The app processes case study PDFs, retrieves relevant context, and interacts with users through a chatbot interface powered by GPT-Neo and FAISS.

## Features

- **PDF Text Extraction**: Upload a case study in PDF format, and the app extracts its content for further processing.
- **Text Chunking**: Breaks large documents into manageable chunks for embedding and retrieval.
- **FAISS Indexing**: Efficiently retrieves the most relevant chunks of information based on user queries.
- **Language Model Integration**: Generates context-aware responses using GPT-Neo (EleutherAI).
- **Interactive Interface**: Engage in a simulated case interview via a Streamlit web app.

## Installation

### Prerequisites

- Python 3.8+
- Pip package manager

### Clone the Repository

```bash
git clone https://github.com/NikoHems/RAG_Chatbot.git
cd RAG_Chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
```

## Usage

### Run the App

```bash
streamlit run app.py
```

### Upload a Case Study

1. Click the "Browse files" button to upload a PDF containing a case study.
2. Click "Process PDF" to extract and index the text.

### Engage in the Interview

1. Type your response to the interviewer's question in the input box.
2. Click "Send" to submit your response.
3. Review the chatbot's reply in the conversation history.

## Project Structure

```
RAG_Chatbot/
├── app.py                 # Main Streamlit application file
├── requirements.txt       # List of dependencies
└── README.md              # Documentation
```

## Key Components

### app.py

- Text Extraction: Extracts text from PDF files using pdfplumber.
- Text Chunking: Preprocesses and splits text into chunks using NLTK's sentence tokenizer.
- FAISS Index: Creates an index for fast similarity search.
- Language Model: Utilizes GPT-Neo to generate responses.
- Streamlit Interface: Provides an interactive UI for uploading PDFs and chatting.

### requirements.txt

Contains the necessary Python libraries, including:

- streamlit
- pdfplumber
- nltk
- faiss
- sentence-transformers
- transformers
- torch

## Model Details

- **SentenceTransformer**: Pre-trained model all-MiniLM-L6-v2 for embedding text chunks.
- **GPT-Neo**: Pre-trained language model EleutherAI/gpt-neo-2.7B for conversational response generation.

## Known Limitations

- Processing large PDFs may take additional time.
- Context retrieval is limited to top-ranked chunks.
- Response quality depends on the language model's pre-training.

## Future Improvements

- Support for multi-file uploads.
- Enhanced prompt engineering for improved conversational accuracy.
- Deployment options for scalability (e.g., Docker, cloud platforms).
- Use of big LLM´s like GPT-4 (although out of project scope for now)

## Contributing

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature-name'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
