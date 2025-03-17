import streamlit as st
import torch
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append(r"C:\Users\Rethenya C R\finalyrproject\yolov5")  
from yolov5 import detect

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [2, 5, 5, 1, 3] 
print(model)

st.title("Real-Time Traffic Detection and Counting")

# Initialize variables for counting
car_count, bus_count,truck_count, pedestrians, two_wheeler = 0, 0, 0, 0, 0
total_in, total_out = 0, 0

# Function to count objects
def count_objects(results):
    global car_count, bus_count, truck_count
    detected_data = results.pandas().xyxy[0]
    car_count = len(detected_data[detected_data['name'] == 'car'])
    bus_count = len(detected_data[detected_data['name'] == 'bus'])
    truck_count = len(detected_data[detected_data['name'] == 'truck'])
    pedestrians = len(detected_data[detected_data['name'] == 'person'])
    two_wheeler = len(detected_data[detected_data['name'] == 'bicycle']) + len(detected_data[detected_data['name'] == 'motorcycle'])
    return car_count, bus_count,truck_count, pedestrians, two_wheeler

# Sidebar settings
st.sidebar.title("Settings")
input_type = st.sidebar.selectbox("Select input type", ["Image", "Video", "Live Webcam"])

if input_type == "Image":
    print("aaaaaaaaaaaaaaaaaaaaaa")
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Perform detection
        results = model(img)
        car_count, bus_count, truck_count, pedestrians, two_wheeler = count_objects(results)
        print(car_count,bus_count,truck_count,two_wheeler,"bbbbbbbbbbbbbbbbbbbbbbbb")

        # Display detection

        st.image(np.squeeze(results.render()), caption="Processed Image", use_column_width=True)
        st.write(f"Detected Cars: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Pedestrian: {pedestrians}, Two-wheeler:{two_wheeler}")
        
        total_vehicles = car_count + bus_count + truck_count + two_wheeler
        print(total_vehicles,"cccccccccc")
        if total_vehicles > 10:  # Example threshold, adjust as needed
            traffic_status = "High Traffic"
        else:
            traffic_status = "Low Traffic"

        # Display traffic level
        print(traffic_status,"ddddddddddd")
        st.write(f"Traffic Level: {traffic_status}")
            

elif input_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        video_path = f"./temp_{uploaded_video.name}"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        cap = cv2.VideoCapture(video_path)

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = model(frame)
            car_count, bus_count, truck_count, pedestrians, two_wheeler = count_objects(results)
            
            # Display frame
            stframe.image(np.squeeze(results.render()), channels="BGR", use_column_width=True)
            st.write(f"Detected Cars: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Pedestrian: {pedestrians}, Two-wheeler:{two_wheeler}")
        cap.release()

elif input_type == "Live Webcam":
    st.write("Starting live webcam feed...")
    cap = cv2.VideoCapture(0)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform detection
        results = model(frame)
        car_count, bus_count, truck_count, pedestrians, two_wheeler = count_objects(results)
        
        # Display frame
        stframe.image(np.squeeze(results.render()), channels="BGR", use_column_width=True)
        st.write(f"Detected Cars: {car_count}, Bus: {bus_count}, Truck: {truck_count}, Pedestrian: {pedestrians}, Two-wheeler:{two_wheeler}")

        # Press 'q' to break the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

st.sidebar.write("Total Vehicles In:", total_in)
st.sidebar.write("Total Vehicles Out:", total_out)