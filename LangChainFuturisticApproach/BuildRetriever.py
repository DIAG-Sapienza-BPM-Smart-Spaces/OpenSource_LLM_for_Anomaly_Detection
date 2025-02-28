import io
import re

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import base64
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

class BuildRetriever():
    
    def __init__(self, retriever, llm, reranker_model="BAAI/bge-reranker-large",  top_k=10):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k  # Number of final ranked chunks to use
        self.reranker = CrossEncoder(reranker_model)  # Load the reranker model

    def rerank_with_bge(self, input_dict):
        """
        Use BGE-Reranker to score and reorder retrieved chunks.
        """
        query = input_dict["question"]
        retrieved_docs = input_dict["retrieved_docs"]
        
        if not retrieved_docs:
            return {"context": [], "question":query}  # Ensure empty list structure is maintained

        
        # Create (query, document) pairs
        pairs = []
        for doc in retrieved_docs:
            if isinstance(doc, Document):
                pairs.append((query, doc.page_content)) 
            else:
                pairs.append((query, doc))
        
        # Compute relevance scores
        scores = self.reranker.predict(pairs)

        # Sort documents by highest relevance score
        ranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        print(ranked_docs[:5])
        # Return only the top-ranked documents
        return {"context": ranked_docs[:self.top_k], "question": query}  # Format correctly for next step

    def split_image_text_types(self, input_dict):
        """
        Extract and format text from retrieved documents.
        """
        docs = input_dict["context"]
        texts = []
        for (doc,score) in docs:
            if isinstance(doc,Document):
                texts.append(doc.page_content)
            else:
                texts.append(doc)
        return {"texts": texts, "question": input_dict["question"]}


    def img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        formatted_texts = "\n".join(data_dict["texts"])
        messages = []

        # print("Formatted Texts: ", data_dict["texts"])

        # Adding the text for analysis
        # text_message_gpt = {
        #     "type": "text",
        #     "text": (
        #         "You are an AI assistant helping users retrieve technical information.\n"
        #         "You will be given a mix of text and tables.\n\n"
        #         f"{formatted_texts}"
        #         "\n\nIn text you may encounter Page number: # to refer to the page number associated with the text.\n"
        #         "In text you may encounter Images on this page: imagePath, imagePath, imagePath to refer to the images in such page.\n"
        #         "You can also insert in the answer the imagepath of the images if present.\n\n"
        #         "📖 **Extract information STRICTLY from the provided context.**\n"
        #         f"User-provided question: {data_dict['question']}\n\n"
        #     ),
        # }

        text_message_qwen = {
            "You are an AI assistant helping users retrieve technical information.\n"
            "You will be given a mix of text and tables.\n\n"
            f"{formatted_texts}"
            "\n\nIn text you may encounter 'Page number: #' to refer to the page number associated with the text.\n"
            "In text you may encounter 'Images on this page: imagePath, imagePath, imagePath' to refer to the images in such page.\n"
            "You have to insert in the answer the imagepath of the images if present.\n\n"
            "📖 **Extract information STRICTLY from the provided context.**\n"
            f"User-provided question: {data_dict['question']}\n\n"
        }

        # messages.append(text_message_gpt)
        # return [HumanMessage(content=messages)]

        return [HumanMessage(content=text_message_qwen)]


    def multi_modal_rag_chain(self):
        """
        Multi-modal RAG chain
        """

        # Multi-modal LLM

        # RAG pipeline
        chain = (
            {
                "retrieved_docs": self.retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.rerank_with_bge)  # ✅ Pass retrieved docs + question to reranker
            | RunnableLambda(self.split_image_text_types)  # ✅ Extract clean text
            | RunnableLambda(self.img_prompt_func)  # ✅ Format into prompt
            | self.llm  # ✅ Pass to LLM
            | StrOutputParser()  # ✅ Parse output
        )
        # chain = (
        #     {
        #         "context": self.retriever | RunnableLambda(self.split_image_text_types),
        #         "question": RunnablePassthrough(),
        #     }
        #     | RunnableLambda(self.img_prompt_func)
        #     | self.llm
        #     | StrOutputParser()
        # )
        return chain