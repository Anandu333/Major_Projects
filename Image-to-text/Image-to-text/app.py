import streamlit as st
import cv2
from PIL import Image
import easyocr
import pandas as pd
from spellchecker import SpellChecker
import tempfile
import os

def main():
    # Title and description
    st.title("Image to Text with Text Correction")
    st.write("Upload an image and get the text with spelling correction!")

    # Define directory to store uploaded images
    upload_dir = "uploaded_images"
    os.makedirs(upload_dir, exist_ok=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # Store the uploaded file in the specified directory
        image_path = os.path.join(upload_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

# Display full path of the uploaded image

        full_image_path = os.path.abspath(image_path)

    #Passing full image to our easyocr module
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(full_image_path)

    # coverting our text into data frame to better understand
        df = pd.DataFrame(result, columns=['Cor', 'text', 'confidence'])
        s = [str(x) for x in df["text"]]
        s1 = " ".join(s)


        st.title("Extracted Text from the image: ")
        st.code(s1)

    # spellijg checker
        spell = SpellChecker()
        required_output = ""
        for word in s1.split():
            required_output = required_output + " " + spell.correction(word)

        st.title("Spelling Corrected text after Extraction: ")
        st.code(required_output)

if __name__ == "__main__":
    main()