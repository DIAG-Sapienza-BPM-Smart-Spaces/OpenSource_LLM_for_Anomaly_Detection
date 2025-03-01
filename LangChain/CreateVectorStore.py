import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pickle

class CreateVectorStore():
    def __init__(self, vectorstore, texts, tables, docstore_path):
        self.vectorstore = vectorstore
        self.texts = texts
        self.tables = tables
        self.docstore_path = docstore_path

    def create_multi_vector_retriever(self):
        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=id_key,
        )


        if self.texts:
            self.add_documents(retriever, self.texts, id_key, "text")
            print("✅ Texts added")
            
        if self.tables:
            self.add_documents(retriever, self.tables, id_key, "table") 
            print("✅ Tables added")

        with open(self.docstore_path, "wb") as f:
            pickle.dump(store, f)

        return retriever

    def add_documents(self, retriever, documents, id_key, doc_type):
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        docs = []
        for i, doc in enumerate(documents):
            docs.append(Document(page_content=doc, metadata={id_key: doc_ids[i]}))
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, documents)))
