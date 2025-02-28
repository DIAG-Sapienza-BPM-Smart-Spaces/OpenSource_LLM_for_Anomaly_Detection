import io
import re

from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image
import base64
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

class BuildRetriever():
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def split_image_text_types(self, docs):
        texts = []
        for doc in docs:
            if isinstance(doc, Document):
                doc = doc.page_content
            else:
                texts.append(doc)
        return {"texts": texts}


    def img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []

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
            "\n\nIn text you may encounter Page number: # to refer to the page number associated with the text.\n"
            "In text you may encounter Images on this page: imagePath, imagePath, imagePath to refer to the images in such page.\n"
            "You can also insert in the answer the imagepath of the images if present.\n\n"
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
                "context": self.retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
            | self.llm
            | StrOutputParser()
        )

        return chain