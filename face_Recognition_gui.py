import tkinter as tk
from tkinter import messagebox

import Dataset_Capture
import train_model
import Recognize_face


def run_capture():
    messagebox.showinfo(
        "Instructions",
        "Press ESC to stop capturing"
    )
    Dataset_Capture.capture()


def run_train():
    train_model.train()
    messagebox.showinfo("Done", "Training completed")


def run_recognize():
    Recognize_face.recognize()


root = tk.Tk()
root.title("Face Recognition System")
root.geometry("300x250")

tk.Button(root, text="Capture Faces", width=25, command=run_capture).pack(pady=10)
tk.Button(root, text="Train Model", width=25, command=run_train).pack(pady=10)
tk.Button(root, text="Recognize Face", width=25, command=run_recognize).pack(pady=10)
tk.Button(root, text="Exit", width=25, command=root.quit).pack(pady=10)

root.mainloop()
