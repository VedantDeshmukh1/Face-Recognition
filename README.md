# Face Recognition Folder Protection

This Python script uses face recognition to protect a folder by granting access only when a known face is detected. It captures a dataset of face images using the webcam and then uses the dataset to compare against faces detected in real-time.

## Requirements

- Python 3.x
- OpenCV (cv2)
- face_recognition
- numpy

## Installation

1. Clone the repository or download the script files.

2. Install the required dependencies by running the following command:

   ## Usage

1. Run the script `biometric_new.py` using Python.

2. The script will prompt you to capture a dataset of face images. It will open the webcam and capture a specified number of face images (default is 100). Press 'q' to stop the capture process early.

3. The captured face images will be saved in the specified dataset directory (default is "C:/Users/admin/Desktop/img").

4. After capturing the dataset, the script will start real-time face recognition using the webcam.

5. When a face is detected, the script will compare it against the known faces in the dataset.

6. If a match is found, the script will open the protected folder.

7. If no match is found, access to the folder will be denied.

8. Press 'q' to stop the real-time face recognition and exit the script.

## Configuration

- `dataset_path`: Specify the path where the captured face images will be stored (default is "C:/Users/admin/Desktop/img").
- `num_images`: Specify the number of face images to capture for the dataset (default is 100).

## Notes

- Ensure that the dataset directory exists and has write permissions.
- The script uses the default webcam (index 0) for capturing images and real-time face recognition. If you have multiple webcams, you may need to change the index accordingly.
- The face recognition model used in this script is based on the `face_recognition` library, which uses deep learning for accurate face detection and recognition.
