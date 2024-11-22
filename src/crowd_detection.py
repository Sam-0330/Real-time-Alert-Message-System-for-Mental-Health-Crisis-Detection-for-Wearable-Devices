from datetime import datetime
import cv2
import torch
import numpy as np
import time

from emotions import detect_emotions

def get_datetime():
    now = datetime.now()
    current_datetime_formatted = now.strftime("%Y-%m-%d_%H-%M-%S")
    return current_datetime_formatted 

def detect_crowds():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        start_time = time.time()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  
        output_filename_raw = f'crowd_videos/raw/crowd_raw.{get_datetime()}.avi'
        output_filename_detected = f'crowd_videos/detected/crowd_detected.{get_datetime()}.avi'  
        fps = 20.0 

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width,frame_height)

        raw_out = cv2.VideoWriter(output_filename_raw, fourcc, fps, frame_size)
        detected_out = cv2.VideoWriter(output_filename_detected, fourcc, fps, frame_size)

        while True:
            ret, frame = cap.read()
            raw_out.write(frame)
            if not ret:
                break
            
            results = model(frame)
            result_coordinates = results.xyxy[0].cpu().numpy().tolist()
            frame_people = frame.copy()
            people_count = 0

            for result in result_coordinates:
                class_id = int(result[5])
                if class_id == 0: 
                    people_count += 1
                    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
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
            
            elapsed_time = time.time() - start_time
            if elapsed_time >= 15:
                print(f"State after 10 seconds: {label}")
                cap.release()
                cv2.destroyAllWindows()
                detect_emotions(label,mode="display")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

detect_crowds()
