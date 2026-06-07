# Face Recognition System Using OpenCV

## Description
This project implements a real-time face recognition system using
OpenCV and the LBPH (Local Binary Pattern Histogram) algorithm.
It captures face images, trains a model, and recognizes faces via webcam.

## Technologies Used
- Python 3.x
- OpenCV (opencv-contrib-python)
- NumPy

## Project Structure
- Dataset_Capture.py – Capture face images
- train_model.py – Train LBPH model
- Recognize_face.py – Real-time recognition
- dataset/ – Stored face images
- face_model.yml – Trained model
- labels.npy – Label mapping

## How to Run
1. Capture dataset  
   `python Dataset_Capture.py
2. Train the model  
   `python train_model.py
3. Recognize face  
   `python Recognize_face.py

## Notes
- Lower confidence value indicates better match
- Threshold used: 65
