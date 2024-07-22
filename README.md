# Image-to-Audio-for-the-Visually-Impaired

This project is a web application that generates captions for uploaded images using a deep learning model. The application uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate captions, leveraging the ResNet50 model for image feature extraction and LSTM networks for sequence generation. Additionally, the application provides an audio output of the generated caption using text-to-speech (TTS).

## Features

- Upload an image and receive a descriptive caption.
- Utilizes ResNet50 for image feature extraction.
- Uses LSTM networks for generating captions.
- Provides audio output of the generated caption.
- Web interface built using Flask.

## Requirements

To run this project, you will need the following libraries:

- Flask
- numpy
- keras
- tensorflow
- opencv-python
- pyttsx3
- tqdm
- csv

## Usage

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000`.

3. Upload an image using the provided form and receive the generated caption.

## Code Explanation

### Model Loading

The `load_model` function loads the necessary models and weights for generating captions. It uses ResNet50 for image feature extraction and LSTM networks for caption generation.

### Caption Prediction

The `predict_caption` function takes an image, processes it, and generates a caption using the loaded models. It iteratively predicts the next word in the caption sequence until it reaches the end or the maximum length.

### Flask Application

The Flask application consists of two main routes:

- `/`: The index route that displays the upload form.
- `/after`: The route that handles the image upload, processes the image, generates the caption, and provides an audio output.

### Text-to-Speech

The `speak_text` function uses the `pyttsx3` library to convert the generated caption into speech and play it.

## Acknowledgments

This project utilizes the following models and libraries:

- ResNet50 for image feature extraction.
- LSTM networks for sequence generation.
- Flask for the web interface.
- pyttsx3 for text-to-speech conversion.

