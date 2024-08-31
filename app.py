import streamlit as st
import cv2
import numpy as np
from PIL import Image
import string
import random
import os

# Paths to cascade files
car_cascade_src = 'cascade/cars.xml'
bus_cascade_src = 'cascade/Bus_front.xml'

# Define function to process image
def process_image(uploaded_file):
    # Read image
    image = Image.open(uploaded_file)
    image = image.resize((450, 250))
    image_arr = np.array(image)
    grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)

    # Load cascades
    car_cascade = cv2.CascadeClassifier(car_cascade_src)
    bus_cascade = cv2.CascadeClassifier(bus_cascade_src)

    # Detect objects
    cars = car_cascade.detectMultiScale(grey, 1.1, 1)
    buses = bus_cascade.detectMultiScale(grey, 1.1, 1)

    # Draw rectangles and count objects
    bcnt = len(buses)
    ccnt = 0
    for (x, y, w, h) in buses:
        cv2.rectangle(image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if bcnt == 0:
        for (x, y, w, h) in cars:
            cv2.rectangle(image_arr, (x, y), (x + w, y + h), (255, 0, 0), 2)
            ccnt += 1

    # Convert image array back to image
    result_image = Image.fromarray(image_arr, 'RGB')
    return result_image, ccnt, bcnt

def main():
    st.title("Vehicle Detection App")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")

        # Process image
        result_image, car_count, bus_count = process_image(uploaded_file)

        # Save result image
        result_filename = ''.join(random.choices(string.ascii_lowercase, k=10)) + '.png'
        result_image.save(os.path.join('static/uploads', result_filename))

        # Display result
        st.image(result_image, caption="Processed Image", use_column_width=True)
        st.write(f'{car_count} cars and {bus_count} buses found')
        st.write(f'Saved image: {result_filename}')

if __name__ == "__main__":
    main()
