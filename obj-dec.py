import cv2
import torch
from pathlib import Path

# Load the YOLO model (YOLOv5 in this example)
model = torch.hub.load('ultralytics/yolov5',
                       'yolov5s')  # 'yolov5s' is a lightweight version, you can use other YOLOv5 models as well

# Set up the video input and output
video_path = 'road_trafifc.mp4'  # Path to input video
output_path = 'output_video.mp4'  # Path for saving the output video
video_capture = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Set up the VideoWriter for saving the output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process the video frame by frame
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Perform YOLO object detection
    results = model(frame)

    # Extract detection results
    detections = results.pandas().xyxy[0]  # Get bounding box coordinates and class info

    # Draw bounding boxes and labels on the frame
    for index, row in detections.iterrows():
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence, label = row['confidence'], row['name']

        # Draw bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Display label and confidence
        cv2.putText(frame, f'{label} {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                    2)

    # Show the output frame
    cv2.imshow('YOLO Object Detection', frame)

    # Write frame to output video
    output_video.write(frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
output_video.release()
cv2.destroyAllWindows()
