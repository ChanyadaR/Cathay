import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings  # Import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Retrieve API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set or invalid")

# Initialize the LLM with Groq API
llm = ChatGroq(temperature=0, model="llama3-70b-8192", api_key=groq_api_key)

# Load the PDF files
pdf_loader = PyPDFDirectoryLoader("./doc")
documents = pdf_loader.load()

# Debug: Print the number of documents loaded
print(f"Number of documents loaded: {len(documents)}")

# Split the documents into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Generate embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Create a vector store to store embeddings
vectorstore = FAISS.from_documents(texts, embeddings)

#prompt_template = PromptTemplate(
   #input_variables=["context", "question"],
    #template="You are a helpful assistant. Use the provided context to answer the question in Chinese. The contents in the files are connected, beginning from, UCECF3_0100火險首頁, UCECF3_0200房屋資料, UCECF3_0300保費計算. Find the contents in correct files order\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
#)


#updated prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant. The data in the files is organized sequentially, as follows:\n"
        "1. UCECF3_0100火險首頁\n"
        "2. UCECF3_0200房屋資料\n"
        "3. UCECF3_0300保費計算\n"
        "4. UCECF3_0600火險確認頁\n"
        "5. UCECF3_0800火險資料完成頁\n"
        "6. UCECF3_0900火險續保頁\n\n"
        "Each file builds upon the previous one. You are a helpful assistant. Use this order to interpret and answer questions accurately and use the same language as the question to answer.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
)




# Create a RetrievalQA chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,  # Use the ChatGroq LLM here
    retriever=vectorstore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)


# Function to ask a question
def ask_question(query: str):
    response = retrieval_chain(
        {"query": query}
    )
    return response

# Example usage
#if __name__ == "__main__":
    #user_query = "什麼時候用sessionstorage儲存房屋地址、房屋構造的"
    #response = ask_question(user_query)
    #print("Answer:", response["result"])  # The main answer
    print("Source Documents:", response["source_documents"])  # Optional: source documents


# Example usage
if __name__ == "__main__":
    print("Welcome! You can ask questions about the contents of the PDF files.")
    print("Type 'exit' to quit the program.\n")
    
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        response = ask_question(user_query)
        print("Answer:", response["result"])  # The main answer
        print("Source Documents:", response["source_documents"])  # Optional: source documents