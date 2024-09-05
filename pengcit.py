import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image

# Function to calculate and plot color histogram
def plot_color_histogram(image, ax):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlabel("Intensity Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Color Histogram")
        ax.set_xlim([0, 256])

# Function to perform histogram equalization
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

st.title("Image Processing App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image and its histogram
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, channels="BGR", caption="Uploaded Image")
        
    with col2:
        fig, ax = plt.subplots()
        plot_color_histogram(image, ax)
        st.pyplot(fig)

    # Create expanders
    with st.expander("Histogram Equalization"):
        st.write("Click the button below to perform histogram equalization on the uploaded image.")
        
        if st.button("Equalize Histogram"):
            equalized_image = histogram_equalization(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(equalized_image, channels="BGR", caption="Equalized Image")

            with col2:
                fig, ax = plt.subplots()
                plot_color_histogram(equalized_image, ax)
                st.pyplot(fig)
            
            if st.button("Save Equalized Image"):
                is_success, buffer = cv2.imencode(".jpg", equalized_image)
                if is_success:
                    st.download_button(
                        label="Download Equalized Image",
                        data=BytesIO(buffer).getvalue(),
                        file_name="equalized_image.jpg",
                        mime="image/jpeg"
                    )

    with st.expander("Transformation 2"):
        st.write("This tab can be used for another image transformation.")
        # Implement your second transformation here

    with st.expander("Transformation 3"):
        st.write("This tab can be used for another image transformation.")
        # Implement your third transformation here
