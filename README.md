# ```RAG (Retrieval-Augmented Generation) for the PDF documents```

# Project Outline
    # Data Ingestion:
        -Sourcing document: We will be importing PDF document using streamlit application file uploader.
        -pyPDF library we will use to extract the pdf document into the single string like format.
    # Data preprocessing:
        -Here we are chunking the extracted text into smaller ones using Langchain library RecursiveCharacterTextSplitter. Those chunks will be of fixed sizes and also we need to set the overlap limit.
    # Feature Engineering:
        -Here we are using google pre trained embedding model from google "GoogleGenerativeAIEmbeddings" which converts text chunks into the vector representation. 
    # Pre Trained Model Selection:
        -Here we are going to use Google pretrained model which we will be slecting one of from gemini family. 
    # Vector Database Creation:
        -We will be using FAISS (Facebook AI Similarity Search) vector store data base for our converted vector representations.
    # Model Deployment:
        -Finally, will deploy above code by creating a user interface with the help of streamlit web service.
        -It will have user  


# User must upload PDF, mostly text upto 200mb, Submit and Query the Document.