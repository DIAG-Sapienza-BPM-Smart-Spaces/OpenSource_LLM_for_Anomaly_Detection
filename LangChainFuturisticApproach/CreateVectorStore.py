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
        """
        Create retriever that indexes summaries, but returns raw images or texts
        """

        # Initialize the storage layer
        store = InMemoryStore()
        id_key = "doc_id"

        # Create the multi-vector retriever
        retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": 20}
        )

        # Add texts and tables
        # Check that text_summaries is not empty before adding
        if self.texts:
            self.add_documents(retriever, self.texts, id_key, "text")
            print("✅ Texts added")
        # Check that table_summaries is not empty before adding
        if self.tables:
            self.add_documents(retriever, self.tables, id_key, "table") 
            print("✅ Tables added")

        with open(self.docstore_path+".pkl", "wb") as f:
            pickle.dump(store, f)
        with open(self.docstore_path+".txt", "w") as f:
            f.write(str(store))

        # Add texts, tables, and images
        # Check that text_summaries is not empty before adding
        # if self.text_summaries:
        #     self.add_documents(retriever, self.text_summaries, self.texts, id_key)
        # Check that table_summaries is not empty before adding
        # if self.table_summaries:
        #     self.add_documents(retriever, self.table_summaries, self.tables, id_key)
        # Check that image_summaries is not empty before adding
        # if self.image_summaries:
        #     self.add_documents(retriever, self.image_summaries, self.images, id_key)

        return retriever

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(self, retriever, documents, id_key, doc_type):
        doc_ids = [str(uuid.uuid4()) for _ in documents]
        docs = []
        for i, doc in enumerate(documents):
            docs.append(Document(page_content=doc, metadata={id_key: doc_ids[i]}))
        retriever.vectorstore.add_documents(docs)
        retriever.docstore.mset(list(zip(doc_ids, documents)))

        # with open(f"{doc_type}_docs.pkl", "wb") as f:
        #     pickle.dump(docs, f)
        # with open(f"{doc_type}_doc_ids.pkl", "wb") as f:
        #     pickle.dump(doc_ids, f)