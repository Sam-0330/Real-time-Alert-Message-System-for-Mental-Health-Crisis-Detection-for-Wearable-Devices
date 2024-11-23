# Real-Time Mental Health Crisis Detection System

## Project Objective
The objective of this project is to develop an advanced mental health crisis detection system that integrates environmental context detection, facial emotion recognition, and simulated physiological data (such as heart rate and blood pressure) to monitor an individual’s well-being in real-time. The system aims to identify early signs of emotional distress or potential health emergencies and send timely alerts to ensure proactive intervention.

---

## Software Components Used
1. **Python**: Programming language for implementing the detection algorithms.
2. **YOLOv8**: Object detection model for recognizing environments (crowded/isolated) and identifying patterns.
3. **OpenCV**: For video and image processing.
4. **TensorFlow/PyTorch**: Frameworks used to train and deploy the model.
5. **Matplotlib/Seaborn**: For generating visualizations and data analysis graphs.
6. **SMTP**: For sending alert messages via email.

---

## Hardware Components Used
1. **Mobile Phone**: Used as a camera to capture real-time video and images for analysis.
2. **Standard PC or Laptop**: Required for running the trained YOLOv8 model and processing the data.

---

## Datasets Used
1. **FER 2013 (Facial Emotion Recognition Dataset)**:  
   - A publicly available dataset consisting of labeled images for emotion detection.  
   - Dataset Link: [FER 2013 Dataset on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

2. **COCO (Common Objects in Context Dataset)**:  
   - A large-scale dataset for object detection, used for training the YOLOv8 model to classify environments as crowded or isolated.  
   - Dataset Link: [COCO Dataset](https://cocodataset.org/)

---

## Outcome of the Project
The project successfully demonstrates the following outcomes:
1. **Environment Analysis**: Accurately detects whether the individual is in a crowded or isolated area.
2. **Facial Expression Recognition**: Identifies emotions like fear, anger, or sadness using real-time video data.
3. **Simulated Physiological Data**: Generates blood pressure and heart rate values based on the detected emotions.
4. **Alert System**: Sends timely emergency alerts when abnormal conditions such as high blood pressure or rapid heart rate are detected.
5. **Data Recording**: Captures and stores the user's state, emotional analysis, and physiological readings as a video and structured data for further analysis.

This system provides a comprehensive solution for mental health crisis detection, enabling timely intervention and improving the scope of real-time healthcare monitoring systems.
