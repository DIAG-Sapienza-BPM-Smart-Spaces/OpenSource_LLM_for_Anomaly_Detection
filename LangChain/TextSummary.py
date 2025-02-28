from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
        prompt_text = """\
            You are an AI assistant helping users retrieve technical information.

            📖 **Extract information STRICTLY from the provided context.**
            🚨 **If the answer is not in the context, DO NOT GUESS or invent information.**

            🔎 **Query:**  
            {element}

            📌 **Instructions:**  
            1️⃣ **Extract answers using ONLY the provided context** and focus on accuracy.  
            2️⃣ **Cite the page number(s)** and supporting details from the document where the answer was found.  
            3️⃣ If no relevant answer is found, respond with:  
                `"No relevant information found in the provided context."`

            ⚠️ **Important:** Refer to images only if they are essential for answering the query. Do not describe images unnecessarily.

            **Answer:**"""
        prompt = ChatPromptTemplate.from_template(prompt_text)

        # Text summary chain
        summarize_chain = {"element": lambda x: x} | prompt | self.llm | StrOutputParser()

        # Initialize empty summaries
        text_summaries = []
        table_summaries = []

        # Apply to text if texts are provided and summarization is requested
        if self.texts and self.summarize_texts:
            text_summaries = summarize_chain.batch(self.texts, {"max_concurrency": 5})
        elif self.texts:
            text_summaries = self.texts

        # Apply to tables if tables are provided
        if self.tables:
            table_summaries = summarize_chain.batch(self.tables, {"max_concurrency": 5})

        return text_summaries, table_summaries
