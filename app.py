from flask import Flask, request, jsonify, render_template, redirect
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms.ollama import Ollama
import os
import tempfile
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.schema import Document
from faster_whisper import WhisperModel
from langdetect import detect
from googletrans import Translator
import uvicorn
from asgiref.wsgi import WsgiToAsgi

# Download necessary NLTK data
nltk.download('punkt')  # Download tokenization data
nltk.download('stopwords')  # Download stopwords data

# Initialize Flask application
app = Flask(__name__)  # Create a new Flask application instance

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    words = word_tokenize(text)  # Tokenize text into words
    stop_words = set(stopwords.words('english'))  # Define stop words set
    words = [word for word in words if word not in stop_words]  # Remove stop words
    preprocessed_text = ' '.join(words)  # Join words back into a single string
    return preprocessed_text  # Return preprocessed text

# Initialize session state
messages = []  # List to keep track of conversation messages
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)  # Initialize memory for conversation history

# Embeddings model using HuggingFace's transformer model
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})  # Load embeddings model

# Initialize db_retriever with a default value
db_retriever = None  # Placeholder for the document retriever

# Define the prompt template for LLM
prompt_template = """This is a chat template. As a chat bot, your primary objective is to provide accurate and concise information based on the user's questions about the uploaded document. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the uploaded document. Do not give any other information or note.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
"""  
# Define the prompt template
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])  # Create PromptTemplate instance

# Initialize the LLM model using Ollama (Llama3)
llama_model = Ollama(model="llama3")  # Load Llama3 model

# Translator for language detection and translation
translator = Translator()  # Create a Translator instance

# Route for the homepage
@app.route('/')
def index():
    return render_template('index1.html')  # Render the homepage template

# Route for uploading a PDF and processing it
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global db_retriever  # Use the global variable for db_retriever

    if 'file' not in request.files:  # Check if the 'file' key is in the request
        return redirect(request.url)  # Redirect if no file was found

    file = request.files['file']  # Get the uploaded file

    if file.filename == '':  # Check if the file has no name
        return redirect(request.url)  # Redirect if file name is empty

    if file:  # Proceed if a file is uploaded
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:  # Create a temporary file for PDF
            temp_file.write(file.read())  # Write the uploaded file's contents to the temporary file
            temp_file_path = temp_file.name  # Get the path of the temporary file

        loader = PyPDFLoader(temp_file_path)  # Initialize PDF loader
        documents = loader.load()  # Load documents from the PDF

        preprocessed_documents = [Document(page_content=preprocess_text(doc.page_content), metadata=doc.metadata) for doc in documents]  # Preprocess text in documents

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)  # Initialize text splitter
        texts = text_splitter.split_documents(preprocessed_documents)  # Split documents into chunks

        faiss_db = FAISS.from_documents(texts, embeddings)  # Create FAISS vector store from documents
        faiss_db.save_local("ipc_vector_db")  # Save the FAISS database locally

        db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)  # Load the FAISS database
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})  # Create retriever from FAISS database

        return jsonify({'message': 'PDF processed and database created successfully!'})  # Return success message

# Route for handling text queries
@app.route('/ask_question', methods=['POST'])
def ask_question():
    global db_retriever  # Use the global variable for db_retriever

    if db_retriever is None:  # Check if db_retriever is initialized
        return jsonify({'message': 'Please upload a PDF file to create the database before asking questions.'})  # Return error message if not initialized

    user_input = request.json.get('input')  # Get user input from JSON request

    detected_language = detect(user_input)  # Detect the language of the user input

    messages.append({"role": "user", "content": user_input})  # Append user input to messages

    qa = ConversationalRetrievalChain.from_llm(  # Create ConversationalRetrievalChain instance
        llm=llama_model,  # LLM model
        memory=memory,  # Memory for conversation history
        retriever=db_retriever,  # Document retriever
        combine_docs_chain_kwargs={'prompt': prompt}  # Prompt for combining document chains
    )

    result = qa.invoke(input=user_input)  # Get result from the QA chain
    answer = result["answer"]  # Extract answer from the result

    if detected_language != 'en':  # Check if the detected language is not English
        translated_answer = translator.translate(answer, dest=detected_language).text  # Translate answer to detected language
        response = {  # Prepare response with translation details
            'response': translated_answer,
            'english_response': answer,
            'language': detected_language
        }
    else:
        response = {  # Prepare response without translation
            'response': answer,
            'language': detected_language
        }

    messages.append({"role": "assistant", "content": answer})  # Append assistant's answer to messages

    return jsonify(response)  # Return the response as JSON

# Route to reset the conversation memory and history
@app.route('/reset', methods=['POST'])
def reset():
    global messages, memory  # Use global variables for messages and memory
    messages = []  # Clear messages
    memory.clear()  # Clear memory
    return jsonify({'message': 'Conversation reset successfully!'})  # Return success message

# Route to transcribe an audio file
@app.route('/get_answer_audio', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:  # Check if the 'audio' key is in the request
        return jsonify({'message': 'No audio file uploaded'}), 400  # Return error message if no audio file is uploaded

    audio_file = request.files['audio']  # Get the uploaded audio file
    audio_path = os.path.join(tempfile.gettempdir(), 'audio.mp3')  # Define path for temporary audio file
    audio_file.save(audio_path)  # Save the uploaded audio file

    model_size = "large-v3"  # Define model size for Whisper
    model = WhisperModel(model_size, device="cuda", compute_type="float16")  # Initialize WhisperModel

    segments, info = model.transcribe(audio_path, beam_size=5)  # Transcribe audio file
    transcription = " ".join(segment.text for segment in segments)  # Join transcribed text segments

    return jsonify({'transcription': transcription})  # Return transcription as JSON

# Run the Flask application
if __name__ == '__main__':
    asgi_app = WsgiToAsgi(app)  # Convert WSGI app to ASGI
    uvicorn.run(asgi_app, host="0.0.0.0", port=5010)  # Run the ASGI app with uvicorn