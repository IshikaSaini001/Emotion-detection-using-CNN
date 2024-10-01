# Emotion Detection using CNN

 This projec implements a Convolutional Neural Network (CNN) for detecting human emotions from facial expressions. It uses Python, TensorFlow, Keras, and OpenCV.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training](#model-training)
- [GUI](#gui)
- [Acknowledgments](#acknowledgments)

## Overview

This project uses a deep learning model (CNN) to classify facial emotions. The facial regions are detected using Haar Cascade and then passed through the model to predict the emotion. The emotions detected include happiness, sadness, anger, and surprise, among others.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/IshikaSaini001/emotion-detection-cnn.git
    cd emotion-detection-cnn
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Download the `haarcascade_frontalface_default.xml` file from the repository or [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades) if not included.

## Project Structure

```plaintext
emotion-detection-cnn/
│
├── gui.py                      # Python script to run the emotion detection GUI
├── model_a1.json                # JSON file containing model architecture
├── model_creation.ipynb         # Jupyter Notebook for creating and training the model
├── model_weights1.h5            # Trained model weights
├── mygui.ipynb                  # Jupyter Notebook for building the GUI
├── README.md                    # Project documentation
├── haarcascade_frontalface_default.xml   # Pre-trained XML classifier for face detection
└── requirements.txt             # Required Python libraries
```

- `gui.py`: Contains the Python code for running the graphical user interface (GUI) for emotion detection.
- `haarcascade_frontalface_default.xml`: Haar cascade classifier for detecting faces.
- `model_a1.json`: Contains the architecture of the CNN used for emotion detection.
- `model_creation.ipynb`: Notebook for creating, training, and saving the CNN model.
- `model_weights1.h5`: The trained weights of the CNN model.
- `mygui.ipynb`: Jupyter notebook that builds a simple GUI for emotion detection.
- `README.md`: This file.
  
## Usage

### Running the GUI

To run the emotion detection GUI:

```bash
python gui.py
```

The program will open a window that captures video from your webcam. It will detect faces using the Haar cascade and predict the emotion displayed by the user.

### Model Training

1. Open the `model_creation.ipynb` notebook and run the cells to train the CNN model on the desired dataset.
2. The model architecture is saved in `model_a1.json`, and the weights are saved in `model_weights1.h5`.

## Acknowledgments

- [OpenCV](https://opencv.org/) for the Haar Cascade Classifier.
- [Keras](https://keras.io/) and [TensorFlow](https://www.tensorflow.org/) for deep learning support.
- Inspiration from the [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) for training the emotion detection model.

