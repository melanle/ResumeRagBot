from flask import Flask, render_template, request
import fitz
# from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    
def get_resume_text(pdf_docs):
    text = ""
    for pdf_path in pdf_docs:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the question requires calculation (e.g., total months/years of experience), carefully read all the dates mentioned in the context and calculate accordingly.
    If the answer is not in provided context, just say, "No information regarding this in the resume", don't provide wrong answer\n\n
    Context: \n {context}?\n
    Question: \n {question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature = 0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt, verbose=True)
    return chain

def ask_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    print(response)
    return response['output_text']


@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    uploaded_resume = None
    existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if existing_files:
        uploaded_resume = existing_files[0]

    if request.method == 'POST':
        
        if 'pdf' in request.files:
            
            existing_file = os.listdir(app.config['UPLOAD_FOLDER'])
            if existing_file:
                for filename in existing_file:
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path) 
            
            pdf = request.files.getlist('pdf')
            
            saved_file = []
            
            for file in pdf:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                saved_file.append(file_path)
            
            raw_text = get_resume_text(saved_file)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)

        if 'question' in request.form:
            question = request.form['question']
            response = ask_question(question)

    return render_template('home.html', response=response, uploaded_resume=uploaded_resume)

if __name__ == '__main__':
    app.run(debug=True)