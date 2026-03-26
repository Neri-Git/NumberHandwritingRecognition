import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw

from model import load_model, load_letter_model

CANVAS_SIZE = 280
BRUSH_SIZE = 18


def preprocess_image(img):
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
    def __init__(self, root, digit_model, letter_model):
        self.root = root
        self.root.title("Digit & Letter Recognizer")

        self.digit_model = digit_model
        self.letter_model = letter_model

        self.mode = tk.StringVar(value="digit")

        self.image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        frame = ttk.Frame(root, padding=10)
        frame.grid()

        # Canvas
        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=8)

        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Buttons
        ttk.Button(frame, text="Predict", command=self.predict).grid(row=0, column=1, padx=10)
        ttk.Button(frame, text="Clear", command=self.clear).grid(row=1, column=1, padx=10)

        # Mode selector
        ttk.Label(frame, text="Mode:").grid(row=2, column=1, pady=(10, 0))

        ttk.Radiobutton(frame, text="Digits", variable=self.mode, value="digit").grid(row=3, column=1)
        ttk.Radiobutton(frame, text="Letters", variable=self.mode, value="letter").grid(row=4, column=1)

        # Result
        self.result_label = ttk.Label(frame, text="Prediction: -", font=("Arial", 16))
        self.result_label.grid(row=5, column=1, pady=10)

        # Probabilities
        self.prob_label = tk.Text(frame, width=25, height=10)
        self.prob_label.grid(row=6, column=1, padx=10)

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

        if self.mode.get() == "digit":
            predictions = self.digit_model.predict(processed)[0]
            labels = [str(i) for i in range(10)]
        else:
            predictions = self.letter_model.predict(processed)[0]
            labels = [chr(ord('A') + i) for i in range(26)]

        predicted_index = np.argmax(predictions)
        predicted_label = labels[predicted_index]
        confidence = predictions[predicted_index]

        self.result_label.config(
            text=f"Prediction: {predicted_label} ({confidence * 100:.2f}%)"
        )

        sorted_indices = np.argsort(predictions)[::-1]

        self.prob_label.delete("1.0", tk.END)

        for i in range(5):
            idx = sorted_indices[i]
            self.prob_label.insert(
                tk.END,
                f"{i+1}. {labels[idx]}: {predictions[idx]*100:.2f}%\n"
            )


def main():
    digit_model = load_model()
    letter_model = load_letter_model()

    root = tk.Tk()
    app = DigitApp(root, digit_model, letter_model)
    root.mainloop()


if __name__ == "__main__":
    main()