from PIL import Image
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
import cv2
import io
import numpy as np
import os
import streamlit as st
from spellchecker import SpellChecker
from paddleocr import PaddleOCR

ocr= PaddleOCR(use_angle_cls=True, lang ='en')

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
        img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)


        # Increase contrast
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 100

        # Convert pixel values back to uint8
        imgContrast = np.uint8(imgContrast)

        # Morphological operation: erosion
        kernel = np.ones((3, 3), np.uint8)
        imgMorph = cv2.erode(imgContrast, kernel, iterations=1)

        result = ocr.ocr(img, cls=True)
        tuples_list = []
        for sublist1 in result:
            for sublist2 in sublist1:
                tuples_list.append(sublist2[1])

        # Print the extracted tuples
        string = ""
        for i in tuples_list:
            string += i[0] + " "

        st.title("Extracted Text from the image: ")
        st.code(string)







        # Initialize SpellChecker
        spell = SpellChecker()

        # Split text into lines
        lines = string.strip().split('\n')

        # Initialize variables
        new_lines = []

        # Iterate through each line
        for line in lines:
            # Split each line into words
            words = line.split()
            new_line = ""
            # Iterate through each word
            for word in words:
                # Get the correction for the word
                correction = spell.correction(word)
                # Check if the correction is not None
                if correction is not None:
                    new_line += correction + " "
                else:
                    # If no correction is found, use the original word
                    new_line += word + " "
            # Append the corrected line to the list of new lines
            new_lines.append(new_line.strip())

        corrected_text = '\n'.join(new_lines)

        st.title("Spelling Corrected Text after extraction: ")
        st.code(corrected_text)

if __name__ == "__main__":
    main()
