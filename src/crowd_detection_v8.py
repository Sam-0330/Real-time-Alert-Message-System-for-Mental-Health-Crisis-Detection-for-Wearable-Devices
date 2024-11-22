from datetime import datetime
import cv2
import torch
import numpy as np
import time
from emotions import detect_emotions
from ultralytics import YOLO

def get_datetime():
    now = datetime.now()
    current_datetime_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    return current_datetime_formatted 

def detect_crowds():
    model = YOLO("yolov8n.pt")  
    cap = cv2.VideoCapture(0)  
    
    if cap.isOpened():
        start_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        output_filename_raw = f'crowd_videos/raw/crowd_raw.{get_datetime()}.avi'
        output_filename_detected = f'crowd_videos/detected/crowd_detected.{get_datetime()}.avi'  
        fps = 20.0 

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        raw_out = cv2.VideoWriter(output_filename_raw, fourcc, fps, frame_size)
        detected_out = cv2.VideoWriter(output_filename_detected, fourcc, fps, frame_size)

        while True:
            ret, frame = cap.read()
            raw_out.write(frame)
            if not ret:
                break
            
            results = model(frame)
            result_coordinates = results[0].boxes.xyxy.cpu().numpy().tolist() 
            class_ids = results[0].boxes.cls.cpu().numpy().tolist()  
            confidences = results[0].boxes.conf.cpu().numpy().tolist()  

            frame_people = frame.copy() 
            people_count = 0 

            for i, result in enumerate(result_coordinates):
                print(result, class_ids[i], confidences[i]) 

                x1, y1, x2, y2 = result  
                confidence = confidences[i] 
                class_id = int(class_ids[i]) 

                if confidence > 0.5 and class_id == 0: 
                    people_count += 1
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                    print(f"Person Detected with Confidence: {confidence}")
                    cv2.rectangle(frame_people, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if people_count < 2:
                label = "Isolated"
                color = (0, 255, 0)  
            else:
                label = "Crowded"
                color = (0, 0, 255)  

            cv2.putText(frame_people, f"People: {people_count} - {label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
            detected_out.write(frame_people)
            
            cv2.imshow("People Detection", frame_people)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed_time = time.time() - start_time
            if elapsed_time >= 15:
                print(f"State after 15 seconds: {label}")
                cap.release()
                cv2.destroyAllWindows()
                detect_emotions(label, mode="display")  
                break
        
        cap.release()
        cv2.destroyAllWindows()

detect_crowds()
