# Sign Language Recognition Project

This project is based on the foundation provided by [Kazuhito00's Hand Gesture Recognition using MediaPipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe).

For those interested in the original project and its details, please visit the [original repository](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). It is a comprehensive implementation; however, note that it uses an older version of TensorFlow.

### Adaptations and Modifications
In this repository, I have developed a Sign Language Recognition system, building upon the work of Kazuhito00. The key adaptations and modifications I have made include:

1. **Removed Point History Classification and Pointer Motion Recognition**: These components were unnecessary for my current project scope.
   
2. **Switched from HDF5 to Keras Format**: Due to compatibility issues with newer versions of TensorFlow, I migrated from HDF5 to Keras format for improved interoperability.

3. **Added Concrete Function for TFL Conversion**: Implemented a Concrete Function to facilitate the conversion of TensorFlow models to TensorFlow Lite (TFL) format.

4. **Customized Hands Keypoints Skeleton**: Enhanced the visualization of hand keypoints to improve interpretability and usability.

5. **Rebuilt Dataset for Letters Recognition**: Reconstructed the dataset specifically tailored for recognizing sign language letters.

# Getting Started
To get started with this project:
* Clone this repository.
* Install the necessary dependencies as outlined.

### Requirements
* mediapipe 0.10.14
* OpenCV 4.10.0.82
* Tensorflow 2.16.1
* scikit-learn 1.5.0 (Only if you want to display the confusion matrix)
* matplotlib 3.9.0 Later (Only if you want to display the confusion matrix)

### Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│    │  keypoint.csv
│    │  keypoint_classifier.py
│    │  keypoint_classifier.tflite
│    └─ keypoint_classifier_label.csv
│ 
└─utils
    └─cvfpscalc.py
</pre>

### app.py
This is a sample program for inference.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign recognition can add and change training data and retrain the model.

# Future Work
- Incorporate additional gestures
- Improve accuracy through fine-tuning.
- Explore real-time applications and optimizations for deployment.

# Acknowledgments
I extend my gratitude to [Kazuhito00](https://github.com/Kazuhito00) for laying the foundation and providing valuable insights into hand gesture recognition using MediaPipe.
