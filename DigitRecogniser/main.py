import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw

from model import load_model

CANVAS_SIZE = 280
BRUSH_SIZE = 18


def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")

    # Resize to MNIST size
    img = img.resize((28, 28))

    # Normalize
    arr = np.array(img) / 255.0

    # Reshape for CNN
    arr = arr.reshape(1, 28, 28, 1)

    return arr


class DigitApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.root.title("Digit Recognizer")

        # Drawing image (for processing)
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        # Layout
        frame = ttk.Frame(root, padding=10)
        frame.grid()

        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=4)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        ttk.Button(frame, text="Predict", command=self.predict).grid(row=0, column=1, padx=10)
        ttk.Button(frame, text="Clear", command=self.clear).grid(row=1, column=1, padx=10)

        self.result_label = ttk.Label(frame, text="Prediction: -", font=("Arial", 16))
        self.result_label.grid(row=2, column=1, padx=10)

        self.prob_label = tk.Text(frame, width=25, height=10)
        self.prob_label.grid(row=3, column=1, padx=10)

    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        if self.last_x is not None:
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                fill="white",
                width=BRUSH_SIZE,
                capstyle=tk.ROUND,
                smooth=True
            )

            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="white",
                width=BRUSH_SIZE
            )

        self.last_x = event.x
        self.last_y = event.y

    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(text="Prediction: -")
        self.prob_label.delete("1.0", tk.END)

    def predict(self):
        processed = preprocess_image(self.image)

        predictions = self.model.predict(processed)[0]

        # Best prediction
        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit]

        self.result_label.config(
            text=f"Prediction: {predicted_digit} ({confidence * 100:.2f}%)"
        )

        # Show top probabilities
        sorted_indices = np.argsort(predictions)[::-1]

        self.prob_label.delete("1.0", tk.END)

        for i in range(5):
            digit = sorted_indices[i]
            prob = predictions[digit]

            self.prob_label.insert(
                tk.END,
                f"{i+1}. {digit}: {prob * 100:.2f}%\n"
            )


def main():
    model = load_model()

    root = tk.Tk()
    app = DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()