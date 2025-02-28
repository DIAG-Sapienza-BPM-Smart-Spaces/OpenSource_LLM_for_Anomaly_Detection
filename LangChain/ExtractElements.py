from langchain_text_splitters import CharacterTextSplitter
from unstructured.partition.pdf import partition_pdf
import os
from itertools import groupby

class ExtractElements():
    # Extract elements from PDF
    def __init__(self, path, fname, lang_chain_path):
        self.path = path
        self.fname = fname
        self.lang_chain_path = lang_chain_path
        self.page_images = {}  # Dictionary to store images per page

    def chunk_by_page(self, elements):
        def get_page_number(element):
            return element.metadata.page_number
        return [list(group) for _, group in groupby(elements, key=get_page_number)]



    def extract_pdf_elements(self):
        """
        Extract images, tables, and chunk text from a PDF file.
        path: File path, which is used to dump images (.jpg)
        fname: File name
        """
        elements =  partition_pdf(
            filename=os.path.join(self.path, self.fname),
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4800,
            new_after_n_chars=4500,
            combine_text_under_n_chars=2250,
            extract_image_block_to_payload=False, 
            extract_image_block_output_dir=self.lang_chain_path,
            include_page_breaks=True,
            extract_image_block_types=["Image"]
        )

        # elements = self.chunk_by_page(elements)
        tables = []
        texts = []

        count_images = 0
        for element in elements:
            page_num  = element.metadata.page_number
            new_element = True
            if "unstructured.documents.elements.CompositeElement" in str(type(element)):
                text_content = str(element)
                text_content+= "\n\n Page number: " + str(page_num)
                for elementInComposite in element.metadata.orig_elements:
                    if "unstructured.documents.elements.Image" in str(type(elementInComposite)):
                        count_images+=1
                        # print(page_num)
                        # print(temp_page_num)
                        # print("images in page" + str(count_images_in_page))
                        figure_path = "figure-" + str(page_num) + "-" + str(count_images) + ".jpg"
                        if(new_element):
                            text_content+= "\n\n Images on this page:\n " + f"{self.lang_chain_path}" + figure_path
                            new_element = False
                        else:
                            text_content+= "\n " + f"{self.lang_chain_path}" + figure_path
                texts.append(text_content)
            elif "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
                
            # if isinstance(element, partition_pdf.__annotations__.get('Image', object)):
            #     page_num = element.metadata.page_number 
            #     image_name = element.metadata
            #     print(element.metadata)
            #     image_path = os.path.join(self.lang_chain_path, image_name)
                
            #     if page_num is not None:
            #         if page_num not in self.page_images:
            #             self.page_images[page_num] = []
            #         self.page_images[page_num].append(image_path)

        return texts, tables