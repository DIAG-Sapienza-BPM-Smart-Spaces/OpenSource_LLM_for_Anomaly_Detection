from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import re

class TextSummary():

    def __init__(self, texts, tables, llm, summarize_texts = False):
        self.texts = texts
        self.tables = tables
        self.summarize_texts = summarize_texts
        self.llm = llm

    # Generate summaries of text elements
    def generate_text_summaries(self):
        """
        Summarize text elements
        texts: List of str
        tables: List of str
        summarize_texts: Bool to summarize texts
        """

        # Prompt
        prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval in short possible way. \
            These summaries will be embedded and used to retrieve the raw text or table elements. \
            Give a concise summary of the table or text that is well optimized for retrieval while keeping all important infos. Table or text: {element} """
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        summarize_chain = {"element": lambda x: x} | prompt | self.llm | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if self.texts and self.summarize_texts: 
            text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 5})
            for doc in text_summaries:
                doc = re.sub(r"<think>.*?</think>\n?", "",doc, flags=re.DOTALL)
        elif self.texts:
            text_summaries = self.texts

        # Apply to tables if tables are provided
        if self.tables:
            table_summaries = summarize_chain.batch(self.tables, {"max_concurrency": 5})

        return text_summaries, table_summaries
