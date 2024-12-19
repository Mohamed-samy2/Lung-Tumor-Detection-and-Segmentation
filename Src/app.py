import streamlit as st
from Models import  MainModel
import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image as PILImage
import io
import os

model =  MainModel()

st.title("Lung Tumor Detection and Segmentation")

uploaded_file = st.file_uploader("Upload Scan Image ", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    print(image.shape)
    
    annotated_image ,pred , h , w , bounding_boxes ,tumor_count = model.run(image)
    
    st.subheader("Results")
    col1, col2, col3 = st.columns(3)

    # Display the original image in the first column
    with col1:
        st.image(image, caption="Original Image", use_container_width=True, channels="GRAY")

    # Display the detected image (annotated) in the second column
    with col2:
        st.image(annotated_image, caption="Detected Image", use_container_width=True,channels="GRAY")

    # Display the mask image (prediction map) in the third column
    with col3:
        overlay_image = cv2.addWeighted(image,0.5, pred.astype(np.uint8)*255, 1, 1)  # Adjust alpha and beta as needed
        st.image(overlay_image, caption="Image with Mask", use_container_width=True, channels="GRAY")
    
    # Display additional results
    st.subheader("Results")
    st.write(f"Tumor Dimensions: {h} x {w}")
    st.write(f"Number of Tumors Detected: {tumor_count}")
    st.write(f"Bounding Boxes: {bounding_boxes}")
    buffer = io.BytesIO()
    
    # Create a ReportLab PDF canvas
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 18)
    title = "Lung Tumor Detection and Segmentation Report"
    title_width = c.stringWidth(title, "Helvetica-Bold", 18)
    c.drawString((width - title_width) / 2, height - 50, title)

    # Add the original image to the PDF
    c.setFont("Helvetica-Bold", 12)
    c.drawString(60, height - 110, "Original Image")
    
    pil_image = PILImage.fromarray(image)
    pil_image.save("original_image.jpg")
    c.drawImage("original_image.jpg", 20, height - 300, width=180, height=180)

    # Add the annotated image to the PDF
    c.setFont("Helvetica-Bold", 12)
    c.drawString(260, height - 110, "Detected Image")
    
    pil_annotated_image = PILImage.fromarray(annotated_image)
    pil_annotated_image.save("annotated_image.jpg")
    c.drawImage("annotated_image.jpg", 220, height - 300, width=180, height=180)

    # Add the overlay image to the PDF
    c.setFont("Helvetica-Bold", 12)
    c.drawString(460, height - 110, "Image with Mask")
    
    pil_overlay_image = PILImage.fromarray(overlay_image)
    pil_overlay_image.save("overlay_image.jpg")
    c.drawImage("overlay_image.jpg", 420, height - 300, width=180, height=180)

    # Add text results
    c.setFont("Helvetica-Bold", 14)
    y_position = height - 350  # Start below the images

    c.drawString(50, y_position, f"Tumor Dimensions: Height {h} x Width {w}")
    y_position -= 20
    c.drawString(50, y_position, f"Number of Tumors Detected: {tumor_count}")
    y_position -= 20
    c.drawString(50, y_position, f"Bounding Boxes: {bounding_boxes}")

    c.save()
    # Move the buffer position to the beginning
    buffer.seek(0)

    os.remove("original_image.jpg")
    os.remove("annotated_image.jpg")
    os.remove("overlay_image.jpg")
    
    # Save the PDF to the disk or provide download link
    st.download_button(
        label="Download PDF Report",
        data=buffer,
        file_name="tumor_detection_report.pdf",
        mime="application/pdf"
    )
