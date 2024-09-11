
# Conversational Document Retrieval System

Welcome to the **Conversational Document Retrieval System**! This project leverages modern natural language processing techniques to enable conversational interactions with documents. Built using Flask and various powerful libraries, this system supports document uploads, text queries, and audio transcriptions. 

## Features

- **Upload PDF Documents**: Upload and process PDF files to create a searchable vector database.
- **Ask Questions**: Interact with the uploaded document using natural language questions.
- **Multilingual Support**: Detects and translates responses based on user input language.
- **Audio Transcription**: Upload audio files to get transcriptions in text form.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Ensure you have the following installed:
- Python 3.7+
- pip (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kashish1344/Conversational-Document-Retrieval-System.git
   cd conversational-document-retrieval
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Flask Application**

   ```bash
   python app.py
   ```

   The application will run on `http://0.0.0.0:5010`.

2. **Navigate to the Application**

   Open your web browser and go to `http://localhost:5010`. 

### API Endpoints

- **GET `/`**: Home page where you can upload documents and interact with the system.
- **POST `/upload`**: Upload a PDF document for processing.
- **POST `/ask_question`**: Submit a question to the system about the uploaded document.
- **POST `/reset`**: Reset the conversation memory and history.
- **POST `/get_answer_audio`**: Upload an audio file to get its transcription.

### Code Overview

- **Flask Application**: Handles routes and integrates various components.
- **Text Preprocessing**: Converts text to lowercase, tokenizes it, and removes stop words.
- **Document Processing**: Loads PDFs, preprocesses them, splits text into chunks, and creates a searchable vector store using FAISS.
- **Conversational Retrieval**: Uses a conversational retrieval chain with the Ollama model to provide accurate answers based on document context.
- **Multilingual Translation**: Detects the language of user queries and translates responses using Google Translate.
- **Audio Transcription**: Transcribes audio files using the Whisper model.

### Requirements

The project relies on several libraries and frameworks. You can find them listed in `requirements.txt`. Some of the key dependencies include:
- `Flask`
- `langchain_community`
- `nltk`
- `faster_whisper`
- `googletrans`
- `uvicorn`
- `asgi_ref`

