import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
import faiss
import numpy as np
import pickle
from transformers import pipeline

# Initialize Streamlit UI
st.title("News Website Q&A")
st.write("Provide 3 links of any news website and ask a question.")

# Input fields for URLs and query
url1 = st.text_input("Enter the first URL")
url2 = st.text_input("Enter the second URL")
url3 = st.text_input("Enter the third URL")
query = st.text_input("Enter your question")

# Path for the FAISS index file
file_path = "vector_index.pkl"

# Button to start processing
if st.button("Get Answer"):
    if url1 and url2 and url3 and query:
        # Use UnstructuredURLLoader to extract text from URLs
        loader = UnstructuredURLLoader(urls=[url1, url2, url3])
        docs = loader.load()

        # Initialize the HuggingFaceEmbeddings model
        embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

        # Process the content into chunks and create embeddings
        embeddings = embeddings_model.embed_documents([doc.page_content for doc in docs])
        embeddings = np.array(embeddings)  # Convert list to NumPy array

        # Check if the FAISS index exists
        if not os.path.exists(file_path):
            st.write("FAISS index not found. Creating a new FAISS index.")

            # Create FAISS index
            dimension = embeddings.shape[1]  # Embedding size (number of dimensions)
            index = faiss.IndexFlatL2(dimension)  # Index that uses L2 distance (Euclidean distance)

            # Add embeddings to the FAISS index
            index.add(embeddings.astype(np.float32))

            # Save the FAISS index to a file
            with open(file_path, "wb") as f:
                pickle.dump(index, f)

        else:
            # Load the existing FAISS index
            with open(file_path, "rb") as f:
                index = pickle.load(f)

        # Create the FAISS vector store
        docstore = InMemoryDocstore(dict(enumerate(docs)))

        def embed_query(query):
            return embeddings_model.embed_query(query)

        vector_store = FAISS(
            embedding_function=embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id={i: i for i in range(len(docs))}
        )

        # Create a retriever from the vector store
        retriever = vector_store.as_retriever()

        # Retrieve the most relevant documents (limit to top 3 docs)
        relevant_docs = retriever.get_relevant_documents(query)[:3]

        # Extract relevant content
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Use Hugging Face's pipeline for question answering
        nlp = pipeline("question-answering", model="deepset/roberta-base-squad2")

        # Generate the answer
        answer = nlp(question=query, context=context)

        # Display the retrieved answer
        st.write("Answer to your question:")
        st.write(answer['answer'] if answer else "No relevant information found.")

    else:
        st.write("Please provide all 3 URLs and a query.")