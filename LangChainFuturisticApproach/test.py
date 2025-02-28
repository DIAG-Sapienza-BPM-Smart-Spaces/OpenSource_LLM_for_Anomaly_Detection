from ExtractElements import *
from TextSummary import *
from ImageSummary import *
from CreateVectorStore import *
from BuildRetriever import *
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

import pickle
import json
import os
import shutil


# File path
fpath = "../"

# fname = "IPC-simple.pdf"
fname = "IPC-A-610F-10-60.pdf"
# fname = "IPC-A-610F.pdf"

extractElementsFlag = False
retrieverFlag = False


model_name = "deepseek-r1:32b"
# embedding_model_name = "nomic-embed-text"
llm_text = OllamaLLM(model=model_name, temperature=0.0)
llm_image = OllamaLLM(model=model_name, temperature=0.0) #max_tokens 1024
llm_retriever = OllamaLLM(model=model_name, temperature=0.0) #max_tokens 1024
embeddings = OllamaEmbeddings(model=model_name)




content_path = f"{model_name}_content"
if not os.path.exists(content_path):
    os.makedirs(content_path)
    print(f"📂 Created folder: {content_path}")

vectorstore_path = f"./{model_name}_vectorstore"
# if not os.path.exists(vectorstore_path):
#     os.makedirs(vectorstore_path)
#     print(f"📂 Created folder: {vectorstore_path}")

lang_chain_path = f"{model_name}_content/images/"
# The vectorstore to use to index the summaries

if extractElementsFlag:
    if os.path.exists(content_path):
        for filename in os.listdir(content_path):
            file_path = os.path.join(content_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete folder and contents
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print(f"✅ Cleared all contents of {content_path}")
    else:
        print(f"⚠️ Directory {content_path} does not exist.")

if retrieverFlag:
    if os.path.exists(vectorstore_path):
        for filename in os.listdir(vectorstore_path):
            file_path = os.path.join(vectorstore_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete folder and contents
            except Exception as e:
                print(f"❌ Error deleting {file_path}: {e}")
        print(f"✅ Cleared all contents of {vectorstore_path}")
    else:
        print(f"⚠️ Directory {vectorstore_path} does not exist.")

if not os.path.exists(lang_chain_path):
    os.makedirs(lang_chain_path)
    print(f"📂 Created folder: {lang_chain_path}")
else:
    print(f"✅ Folder already exists: {lang_chain_path}")


print("Starting ExtractElements Part...")
if extractElementsFlag:
    extractElements = ExtractElements(fpath, fname, lang_chain_path)


    # Get elements texts and tables
    texts, tables = extractElements.extract_pdf_elements()
    
    # for text in texts:
    #     print(text)
    # Optional: Enforce a specific token size for texts
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=5120, chunk_overlap=512
    )
    # joined_texts = " ".join(texts)
    # texts = text_splitter.split_text(joined_texts)
    final_texts = []
    for (page_num, page_content) in texts.items():
        joined_texts = ""
        images = str(page_content["Images"])
        for (key, value) in page_content.items():
            if(key != "Images"):
                joined_texts+=value
                joined_texts+=" "
        splitted_texts = text_splitter.split_text(joined_texts)
        for text in splitted_texts:
            text+= "\n\n Page number: " + str(page_num)
            text+= "\n\n Images on this page:\n " + images
            final_texts.append(text)

    final_tables = []
    for (page_num, page_content) in tables.items():
        joined_tables = ""
        for table in page_content:
            joined_tables+=table
            joined_tables+=" "
        splitted_tables = text_splitter.split_text(joined_tables)
        for table in splitted_tables:
            table+= "\n\n Page number: " + str(page_num)
            final_tables.append(table)
    
    # textSummary = TextSummary(texts, tables, llm_text, True)

    # # Get text, table summaries
    # text_summaries, table_summaries = textSummary.generate_text_summaries()


    with open(f"{model_name}_content/text.pkl","wb") as f:
        pickle.dump(final_texts,f)
    with open(f"{model_name}_content/tables.pkl","wb") as f:
        pickle.dump(final_tables,f)
    with open(f"{model_name}_content/text.txt","w") as f:
        f.write(str(texts))
    with open(f"{model_name}_content/tables.txt","w") as f:
        f.write(str(tables))

    # with open(f"{model_name}_content/text_summaries.pkl","wb") as f:
    #     pickle.dump(text_summaries,f)
    # with open(f"{model_name}_content/table_summaries.pkl","wb") as f:
    #     pickle.dump(table_summaries,f)
    # with open(f"{model_name}_content/text_summaries.txt","w") as f:
    #     f.write(str(text_summaries))
    # with open(f"{model_name}_content/table_summaries.txt","w") as f:
    #     f.write(str(table_summaries))
    print("✅ Extracted elements")
else:
    with open(f"{model_name}_content/text.pkl","rb") as f:
        final_texts = pickle.load(f)
    with open(f"{model_name}_content/tables.pkl","rb") as f:
        final_tables = pickle.load(f)
    # with open(f"{model_name}_content/text_summaries.pkl","rb") as f:
    #     text_summaries = pickle.load(f)
    # with open(f"{model_name}_content/table_summaries.pkl","rb") as f:
    #     table_summaries = pickle.load(f)

    # text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=5120, chunk_overlap=512
    # )
    # joined_texts = " ".join(texts)
    # texts = text_splitter.split_text(joined_texts)
    print("✅ Loaded elements")

# imageSummary = ImmageSummary(lang_chain_path, llm_image)
# # Image summaries
# img_base64_list, image_summaries = imageSummary.generate_img_summaries()
# with open("image.json","w") as f:
#     f.write(str(image_summaries))

# final_tables = []

# Create vectorstore
vectorstore = Chroma(
    collection_name="mm_rag_cj_blog", embedding_function=embeddings, persist_directory=vectorstore_path
)
print("✅ Created VectorStore")

print("Starting MultiVectorRetriever Part...")
if retrieverFlag:
    docstore_path = f"{model_name}_content/docstore"
    multi_vector_retriever = CreateVectorStore(
        vectorstore,
        final_texts,
        final_tables,
        docstore_path
    ).create_multi_vector_retriever()
    print("✅ Saved MultiVectorRetriever components")

else:
    docstore_path = f"{model_name}_content/docstore"
    with open(docstore_path+".pkl", "rb") as f:
        store = pickle.load(f)
    multi_vector_retriever = MultiVectorRetriever(vectorstore=vectorstore,docstore=store,id_key="doc_id", search_kwargs={"k": 20})
    
    print("✅ Loaded MultiVectorRetriever components")


buildRetriever = BuildRetriever(multi_vector_retriever, llm_retriever)

# # Create RAG chain
chain_multimodal_rag = buildRetriever.multi_modal_rag_chain()

# # # # Check retrieval
query = "Tell me about Jackpost Mounting"
# # docs = multi_vector_retriever.invoke(query, limit=6)

# # # We get 4 docs
# # print(len(docs))
# # print(docs)

# # # # Check retrieval
# # # query = "What are the EV / NTM and NTM rev growth for MongoDB, Cloudflare, and Datadog?"
# # # docs = multi_vector_retriever.invoke(query, limit=6)

# # # # We get 4 docs
# # # len(docs)


# # # We get back relevant images
# # img0 = buildRetriever.plt_img_base64(docs[0])
# ## img1 = buildRetriever.plt_img_base64(img_base64_list[0])
# # for img0 in docs:
# #     try:
# #         # Open the image using PIL
# #         image = Image.open(img0)

# #         # Display in Jupyter Notebook
# #         display(image)
# #         print(f"✅ Displayed Image from: {img0}")
# #     except Exception as e:
# #         print(f"❌ Error loading image {img0}: {e}")

# # # try:
# # #     # Open the image using PIL
# # #     image = Image.open(img1)

# # #     # Display in Jupyter Notebook
# # #     display(image)
# # #     print(f"✅ Displayed Image from: {img1}")
# # # except Exception as e:
# # #     print(f"❌ Error loading image {img1}: {e}")


# # Run RAG chain
cleaned_content = re.sub(r"<think>.*?</think>\n?", "",chain_multimodal_rag.invoke(query), flags=re.DOTALL)
print(cleaned_content)