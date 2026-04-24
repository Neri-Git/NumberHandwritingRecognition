import tkinter as tk
from model import load_model
from gui import DigitApp


def main():
    """Start the app: load model and launch GUI."""
    model = load_model()

    root = tk.Tk()
    app = DigitApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()