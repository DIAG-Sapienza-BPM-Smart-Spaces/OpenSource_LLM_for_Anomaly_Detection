import base64
import os
from io import BytesIO
from PIL import Image

from langchain_core.messages import HumanMessage

class ImmageSummary():
    def __init__(self, fpath, llm):
        self.fpath = fpath
        self.llm = llm

    def encode_image(self, image_path):
        """Getting the base64 string"""
        try:
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("utf-8")
            
            # ✅ Check length
            print(f"🔍 Encoded Image Length: {len(encoded)}")

            return encoded
        except Exception as e:
            print(f"❌ Error encoding image: {e}")
            return None


    def image_summarize(self, img_base64, prompt):
        """Summarize an image using the LLM"""

        try:
            # ✅ Send correct JSON format
            msg = self.llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{img_base64}",
                            },
                        ]
                    )
                ]
            )

            # ✅ Debug the structure
            print(f"🔍 Raw LLM Response: {msg}")

            # ✅ Handle different response formats
            if isinstance(msg, str):  # If it's already a string, return it
                return msg
            
            elif isinstance(msg, dict) and "content" in msg:
                return msg["content"]  # Extract response text

            elif hasattr(msg, "content"):
                return msg.content  # Handle object-based response

            else:
                print("🚨 Warning: Unrecognized response format!")
                return "No valid response from model."

        except Exception as e:
            print(f"❌ Error during image summarization: {e}")
            return "Error processing image."
    

    def verify_base64_image(self, base64_str):
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            image.show()  # Displays the image
            print("✅ Image decoded successfully!")
        except Exception as e:
            print(f"❌ Error decoding image: {e}")


    def generate_img_summaries(self):
        """
        Generate summaries and base64 encoded strings for images
        path: Path to list of .jpg files extracted by Unstructured
        """

        # Store base64 encoded images
        img_base64_list = []

        # Store image summaries
        image_summaries = []

        # Prompt
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
        These summaries will be embedded and used to retrieve the raw image. \
        Give a concise summary of the image that is well optimized for retrieval."""

        # Apply to images
        for img_file in sorted(os.listdir(self.fpath)):
            if img_file.endswith(".jpg"):
                img_path = os.path.join(self.fpath, img_file)
                base64_image = self.encode_image(img_path)
                img_base64_list.append(base64_image)
                # self.verify_base64_image(base64_image)
                image_summaries.append(self.image_summarize(base64_image, prompt))

        return img_base64_list, image_summaries