# User Authentication by Facial Recognition

This project implements a face-based user authentication system using computer vision and deep learning.  
The goal is to explore how facial recognition can be used as an alternative authentication mechanism in practical systems.

---

## Problem Statement
Traditional authentication methods such as passwords are inconvenient and vulnerable to reuse or leakage.  
This project investigates whether facial recognition can be used to authenticate users in a controlled environment.

---

## What I Did
- Designed the overall authentication workflow
- Implemented face detection to localize faces from input images
- Applied facial recognition to compare detected faces with stored user data
- Integrated the detection and recognition pipeline into a working system

---

## Technologies Used
- YOLO (face detection)
- DeepFace (face recognition and embedding comparison)
- Python

---

## Outcome
- The system is able to detect faces and authenticate users by comparing facial features
- The workflow functions as expected for basic authentication scenarios

---

## Notes
This project focuses on system design and integration rather than production-grade security.  
Future improvements could include liveness detection and robustness under varied lighting conditions.
