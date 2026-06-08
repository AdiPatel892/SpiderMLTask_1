from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda , RunnablePassthrough,RunnableParallel, RunnableSequence
import streamlit as st
from timeit import default_timer as timer
from dotenv import load_dotenv

load_dotenv()
print("Directory Loading")
loader = DirectoryLoader(
    path="spider",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)
start = timer()
docs=[]
document = loader.lazy_load()
for page in document:
    docs.append(page)
end = timer()

print(f"Time taken by Loader: {end - start:.6f} seconds")
print("Directory Loaded")
# print(len(docs))
# print(docs[0])
print("Splitting Text")
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=75)

chunks = splitter.split_documents(docs)

# print(len(chunks),type(chunks),chunks[0])
print("Text Splitted into chunks")


print("Embedding and Vector Store")
embeddings = HuggingFaceEmbeddings(
model_name="all-MiniLM-L6-v2",

    encode_kwargs={'batch_size': 32}
)
start = timer()
vectorstore= FAISS.from_documents(chunks, embeddings)
end = timer()

print(f"Time taken by code: {end - start:.6f} seconds")
print(type(vectorstore))
print("Embedding and Vector Store")


print("Retrieving Context")
retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={'k':5,'lambda':0.75})




def ContextMaker(context_docs):
    context_text = "\n\n".join(doc.page_content for doc in context_docs)
    return context_text
print("Context Retrived")



print("Generating Prompt")
prompt = PromptTemplate(
    template='''As an Exptert AI/ML Engineer, you are asked to fully understand the content given below, related to AI/Ml research papers . You have to answer the question from the related content . Do not in any case reply from internet and only reply from the content provided. Reply the given data does not answer your question if the content gives no clear answer to the question asked . \n Content = {content} \n\n Question: {question}
    ''',
    input_variables=["content","question"]
)

print(" Prompt created")





print("Model creation")
model = ChatGoogleGenerativeAI(model= "gemini-2.5-flash")


parallel = RunnableParallel({
    "content": RunnableSequence(retriever,RunnableLambda(ContextMaker)),
    "question": RunnablePassthrough(),
})
chain = parallel | prompt | model

st.title("Spider ML Task-1")
st.divider()
st_query = st.text_input("Ask your Question here")
st_go=st.button("Go")

if st_go or st_query:
    result = chain.invoke(st_query)


    st.write(result.content)