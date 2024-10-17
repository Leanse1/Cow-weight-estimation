import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
import os
import streamlit as st

# Load YOLOv8 models
model1 = YOLO("models/eye.pt")
model2 = YOLO("models/cow.pt")

# Calculate the Euclidean distance between two points
def euclidean(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + abs(pt1[1] - pt2[1])**2)

# Function to visualize instance segmentation and keypoint detection
def visualize_combined_results(img, model1, model2):
    results1 = model1(img, save=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform instance segmentation
    # st.write("Instance segmentation results obtained")

    # Perform keypoint detection
    results2 = model2(img, save=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # st.write("Keypoint detection results obtained")

    dist = None
    dist1 = None
    dist2 = None
    
    # Visualize instance segmentation results
    for result in results1:
        # st.write("Processing instance segmentation result")
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for mask, box, score, cls in zip(masks, boxes, scores, classes):
                mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                img_mask = np.zeros_like(img_rgb)
                img_mask[mask_resized] = [229, 22, 122]  # Orange color for mask

                img_rgb = cv2.addWeighted(img_rgb, 1.0, img_mask, 1.0, 1)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (229, 22, 122), 2)

                pt1 = (x1, y1)
                pt2 = (x2, y2)
                dist = euclidean(pt1, pt2)
                # st.write(f'Euclidean distance between top-left and bottom-right corners: {dist:.2f}')
                
                label = f'{model1.names[int(cls)]} {score:.2f}'
                cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            st.write("No masks found for this image.")

    # Visualize keypoint detection results
    for result in results2:
        # st.write("Processing keypoint detection result")
        if result.keypoints is not None and result.boxes is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            point1 = keypoints[0][1][0], keypoints[0][1][1]
            point2 = keypoints[0][2][0], keypoints[0][2][1]
            point3 = keypoints[0][3][0], keypoints[0][3][1]
            point4 = keypoints[0][4][0], keypoints[0][4][1]
            
            # st.write(f"Point1: {point1}, Point2: {point2}")
            # st.write(f"Point3: {point3}, Point4: {point4}")
            dist1 = euclidean(point1, point2)
            dist2 = euclidean(point3, point4)
            # st.write(f'Euclidean distance between pinbone and shoulderbone: {dist1:.2f}')
            # st.write(f'Euclidean distance between girth top and bottom: {dist2:.2f}')

            for keypoint, box, score, cls in zip(keypoints, boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for kp in keypoint:
                    kp_x, kp_y = int(kp[0]), int(kp[1])
                    cv2.circle(img_rgb, (kp_x, kp_y), 3, (0, 0, 255), -1)

                label = f'{model2.names[int(cls)]} {score:.2f}'
                cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if dist:
        x = 11
        lb = 0.45359237
        dist1cm = (x * dist1) / dist
        dist2cm = (x * dist2) / dist
        
        _weight = (dist1cm * dist2cm * dist2cm * lb) / 300
        # st.write(f"Calculated weight: {_weight}")
        return img_rgb, _weight
    return img_rgb, None

# Streamlit app
st.title("Innova8s Cattle Weight Estimation and Segmentation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    filename = uploaded_file.name  
    # Split the filename into parts
    parts = filename.split('_')
    _weight = parts[2]
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # New image size
    new_width = 1040
    new_height = 640

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))

    # Call the combined visualization function
    processed_image, weight = visualize_combined_results(resized_image, model1, model2)

    # Display the processed image
    st.image(processed_image, caption='Processed Image', use_column_width=True)

    if weight is not None:
        html_string1 = f"<p style='font-size:24px;'>Our Estimated Weight: {weight:.2f} kg</p>"
        st.markdown(html_string1, unsafe_allow_html=True)
        # st.write(f"**Our Estimated Weight: {weight:.2f} kg**")
        html_string2 = f"<p style='font-size:24px;'>Actual Weight: {_weight} kg</p>"
        # # st.write(f"**Actual Weight: {_weight} kg**")
        # st.markdown(html_string2, unsafe_allow_html=True)
        

