import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw

CANVAS_SIZE = 280
BRUSH_SIZE = 18


def preprocess_image(img):
    """Convert drawing to 28x28 normalized model input."""
    img = img.convert("L")
    arr = np.array(img)

    coords = np.column_stack(np.where(arr > 20))

    if coords.size == 0:
        return np.zeros((1, 28, 28, 1))

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = arr[y_min:y_max+1, x_min:x_max+1]

    digit_img = Image.fromarray(digit)
    digit_img = digit_img.resize((20, 20))

    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(digit_img, (4, 4))

    arr = np.array(new_img) / 255.0
    arr = arr.reshape(1, 28, 28, 1)

    return arr


class DigitApp:
    """Tkinter app for drawing digits and showing predictions."""

    def __init__(self, root, model):
        """Initialize UI and bind drawing events."""
        self.root = root
        self.model = model

        self.root.title("Digit Recognizer")

        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        frame = ttk.Frame(root, padding=10)
        frame.grid()

        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black"
        )
        self.canvas.grid(row=0, column=0, rowspan=4)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        ttk.Button(frame, text="Predict", command=self.predict).grid(row=0, column=1)
        ttk.Button(frame, text="Clear", command=self.clear).grid(row=1, column=1)

        self.result_label = ttk.Label(frame, text="Prediction: -", font=("Arial", 16))
        self.result_label.grid(row=2, column=1)

        self.prob_label = tk.Text(frame, width=25, height=10)
        self.prob_label.grid(row=3, column=1)

    def start_draw(self, event):
        """Start drawing stroke."""
        self.last_x = event.x
        self.last_y = event.y

    def draw_line(self, event):
        """Draw stroke on canvas."""
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
        """End drawing stroke."""
        self.last_x = None
        self.last_y = None

    def clear(self):
        """Clear canvas and reset UI."""
        self.canvas.delete("all")
        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.result_label.config(text="Prediction: -")
        self.prob_label.delete("1.0", tk.END)

    def predict(self):
        """Run model prediction and display results."""
        processed = preprocess_image(self.image)

        predictions = self.model.predict(processed)[0]

        predicted_digit = np.argmax(predictions)
        confidence = predictions[predicted_digit]

        self.result_label.config(
            text=f"Prediction: {predicted_digit} ({confidence * 100:.2f}%)"
        )

        sorted_indices = np.argsort(predictions)[::-1]

        self.prob_label.delete("1.0", tk.END)

        for i in range(5):
            digit = sorted_indices[i]
            prob = predictions[digit]

            self.prob_label.insert(
                tk.END,
                f"{i+1}. {digit}: {prob * 100:.2f}%\n"
            )