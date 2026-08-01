from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA


#1. Load the document
loader = TextLoader("data/Sample/data.txt")
document = loader.load()

#2. Split the document into chunks. overlap - to maintain the context
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(document)

#3. Create embeddings for the text chunks - translate each chunk into a math representation (embedding)
embeddings = OpenAIEmbeddings()

#4. Create a vector store from the embeddings
vector_store = Chroma.from_documents(texts, embeddings)

#5. Create a QA chain 
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4o-mini"), chain_type="stuff", retriever=vector_store.as_retriever())

#6. Input the question
question = "What payment methods do you accept?"

result = qa.invoke(question)
print(result)