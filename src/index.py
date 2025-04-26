from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from util import pad_fields, rename_fields
from vectorstore import VECTOR_STORE, drop_collection

import glob

def collect_docs():
    """
    Collecting and Loading documents
    """

    PDF_GLOB = "data/*.pdf"
    pdf_files = glob.glob(PDF_GLOB)
    pdf_docs = []
    for path in pdf_files:
        loader = PyPDFLoader(path)
        pages = loader.load()        
        pdf_docs.extend(pages)      

    pad_fields(pdf_docs)
    rename_fields(pdf_docs)
    return pdf_docs

# Collect documents
all_docs = collect_docs()

# Preprocessing and Indexing
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_documents(all_docs)
print(f"Split {len(chunks)} text chunks")

drop_collection()
added = VECTOR_STORE.add_documents(chunks)
print(f"Added {len(added)} text chunks to vector store")



