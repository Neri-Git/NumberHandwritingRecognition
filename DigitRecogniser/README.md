# Handwritten Digit Recognizer

This is a simple Python project I made to recognize handwritten digits.  
It uses TensorFlow and a small neural network to predict digits from 0 to 9.

The app lets you draw a digit on a canvas, then it tries to guess what number it is.

## What it uses

- Python
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Pillow
- Tkinter

## What it does

- Trains a model on MNIST and EMNIST digit datasets
- Lets you draw a digit in a small desktop app
- Predicts the number you drew
- Shows the confidence for each prediction
- Saves the trained model so it can be reused later

## How it works

1. The model is trained on digit images from MNIST and EMNIST.
2. EMNIST images are corrected for orientation before training
3. You draw a digit on the Tkinter canvas.
4. The drawing gets resized and cleaned up before prediction.
5. The model gives a prediction and confidence scores.

## First Run

If no saved model is found, the program will train one automatically.   
This may take a few minutes depending on your system.


## Project files

- `data.py` — loads and prepares the dataset
- `model.py` — builds, trains, saves, and loads the model
- `gui.py` — the GUI where you draw digits
- `main.py` — application entry point
- `digit_model.keras` — saved model file

## How to run it

Clone the project:

```bash
git clone https://github.com/your-username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer
python main.py
```

## Summary

This project demonstrates a full pipeline:

- data loading and preprocessing
- CNN model training
- model persistence
- interactive GUI for real-time predictions

It is a minimal but complete example of deploying a machine learning model in a desktop application.