import os
import subprocess
import zipfile
import argparse
from collections import deque

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import Input # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def download_dataset():
    dataset_zip = 'fer2013.zip'
    data_csv = os.path.join('data', 'fer2013.csv')
    # Download the dataset from Kaggle if it does not exist
    if not os.path.exists(dataset_zip):
        print("Downloading dataset from Kaggle...")
        try:
            # Use subprocess (ensure that you have your Kaggle API configured)
            subprocess.check_call(['kaggle', 'datasets', 'download', '-d', 'deadskull7/fer2013'])
        except subprocess.CalledProcessError as e:
            print("Error downloading dataset:", e)
            raise
    # Extract dataset if CSV does not exist
    if not os.path.exists(data_csv):
        print("Extracting dataset...")
        os.makedirs('data', exist_ok=True)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall('data')


def load_data():
    data_csv = os.path.join('data', 'fer2013.csv')
    print("Loading dataset from", data_csv)
    data = pd.read_csv(data_csv)
    return data


def preprocess_data(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        # Convert space-separated pixel string to an array of uint8 values
        face = np.array([int(pixel) for pixel in pixel_sequence.split(' ')], dtype='uint8')
        face = face.reshape((width, height))
        faces.append(face)
    faces = np.array(faces)
    # Expand dims to add the channel axis (grayscale)
    faces = np.expand_dims(faces, -1)
    # One-hot encode emotion labels
    emotions = pd.get_dummies(data['emotion']).values
    return faces, emotions


def plot_model_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy plot
    axs[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
    axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc='best')
    # Loss plot
    axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
    axs[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc='best')
    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()


def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001),
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        batch_size=64,
                        epochs=50,
                        validation_data=(X_val, y_val),
                        verbose=1)
    plot_model_history(history)
    model.save_weights('model.weights.h5')
    return model


def display_mode(model):
    # Load weights; ensure 'model.weights.h5' exists
    model.load_weights('model.weights.h5')
    # Disable OpenCL usage to reduce logging and potential overhead
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }
    # Map emotion labels to image paths
    emotion_images = {
        "Angry": "data/img/angry.png",
        "Disgusted": "data/img/disgusted.png",
        "Fearful": "data/img/fearful.png",
        "Happy": "data/img/happy.png",
        "Neutral": "data/img/neutral.png",
        "Sad": "data/img/sad.png",
        "Surprised": "data/img/surprised.png",
        "Unknown": "data/img/unknown.png"
    }
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Initialize Haar Cascade once
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Use a deque to smooth predictions
    prediction_buffer = deque(maxlen=10)
    avg_prediction = None  # initialize outside loop

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # Resize ROI to (48, 48) and add batch and channel dimensions
            cropped_img = cv2.resize(roi_gray, (48, 48))
            cropped_img = cropped_img.astype('float32') / 255.0
            cropped_img = np.expand_dims(cropped_img, axis=(0, -1))
            prediction = model.predict(cropped_img)
            prediction_buffer.append(prediction)
            avg_prediction = np.mean(prediction_buffer, axis=0)
            max_index = int(np.argmax(avg_prediction))
            emotion = emotion_dict.get(max_index, "Unknown")

            # Overlay corresponding emotion image if available
            image_path = emotion_images.get(emotion, "data/img/unknown.png")
            emotion_img = cv2.imread(image_path)
            if emotion_img is not None:
                resized_emotion_img = cv2.resize(emotion_img, (w, h))
                frame[y:y+h, x:x+w] = resized_emotion_img

            cv2.putText(frame, emotion, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # If predictions are available, display probabilities in the top left corner
        if avg_prediction is not None:
            prob_text = "Probs: "
            for emo, prob in zip(emotion_dict.values(), avg_prediction[0]):
                prob_text += f"{emo}: {prob*100:.2f}%  "
            cv2.putText(frame, prob_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FER model: train or display")
    parser.add_argument("--mode", help="train/display", default="display")
    args = parser.parse_args()
    mode = args.mode

    # Download and extract dataset if needed
    download_dataset()

    data = load_data()
    faces, emotions = preprocess_data(data)
    faces = faces.astype('float32') / 255.0

    # Split data for training
    X_train, X_val, y_train, y_val = train_test_split(faces, emotions, test_size=0.2, random_state=42)

    # Build model
    model = build_model(input_shape=(48, 48, 1), num_classes=7)

    if mode.lower() == "train":
        train_model(model, X_train, y_train, X_val, y_val)
    elif mode.lower() == "display":
        display_mode(model)
    else:
        print("Invalid mode. Use '--mode train' or '--mode display'.")


if __name__ == '__main__':
    main()
