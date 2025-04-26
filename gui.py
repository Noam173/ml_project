from pathlib import Path
from tkinter import filedialog, messagebox
import customtkinter as ctk
from Data_Manipulation import *
from predict_images import predict_image


def select_and_split() -> None:
    global output_path
    output_path = filedialog.askdirectory()
    path_entry.delete(0, "end")
    path_entry.insert(0, output_path)
    split_data(path=output_path)
    output_path = Path(output_path).parent


def evaluate_train_model() -> None:
    global output_path, model_path, exist
    try:
        classes = output_path / "classes"
        preprocess_data(classes_path=classes, model_path=model_path, exist=exist)
    except Exception:
        messagebox.showerror(
            "Error", "Please select a training path and/or a valid model."
        )
        return


def model() -> None:
    global model_path, exist
    flag = messagebox.askquestion("Model Availability", "Do you have a model?")
    if flag == "yes":
        model_path = filedialog.askopenfilename(
            filetypes=[("Keras Model Files", "*.keras")]
        )
        model_entry.delete(0, "end")
        model_entry.insert(0, model_path)
        exist = True
    else:
        model_path = ""
        model_entry.delete(0, "end")
        model_entry.insert(0, 'No model selected. Press "Load Model" to train.')
        exist = False


def pred_images() -> None:
    global model_path, exist
    if not exist:
        messagebox.showerror("Model Error", "Please select a valid model path.")
        return
    flag = messagebox.askquestion("Folder Selection", "Load a folder?")
    if flag == "yes":
        img_dir = f"{filedialog.askdirectory()}/*"
    else:
        img_dir = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
    predict_image(img_dir=img_dir, model=model_path)


# ========== GUI STYLING ==========

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")  # You can change to "green", "dark-blue", etc.

window = ctk.CTk()
window.title("Noam's Project GUI")
window.geometry("900x550")
window.resizable(False, False)
window.configure(bg="#121212")  # Darker background

frame = ctk.CTkFrame(window, corner_radius=15, fg_color="#1E1E1E")
frame.pack(fill="both", expand=True, padx=30, pady=30)

title = ctk.CTkLabel(
    frame,
    text="Welcome to Noam's Machine Learning Project",
    font=("Arial Rounded MT Bold", 24),
    text_color="white",
)
title.pack(pady=(20, 10))

path_entry = ctk.CTkEntry(
    frame,
    placeholder_text="Load and split the data",
    corner_radius=10,
    width=600,
    font=("Arial", 14),
)
path_entry.pack(pady=10)

model_entry = ctk.CTkEntry(
    frame,
    placeholder_text="Model's file path",
    corner_radius=10,
    width=600,
    font=("Arial", 14),
)
model_entry.pack(pady=10)

button_frame = ctk.CTkFrame(frame, fg_color="transparent")
button_frame.pack(pady=20)

path_button = ctk.CTkButton(
    button_frame,
    text="Select Data Path",
    command=select_and_split,
    corner_radius=10,
    width=200,
    hover_color="purple",
)
path_button.grid(row=0, column=0, padx=10)

model_button = ctk.CTkButton(
    button_frame,
    text="Load Model",
    command=model,
    corner_radius=10,
    width=200,
    hover_color="purple",
)
model_button.grid(row=0, column=1, padx=10)

predict_button = ctk.CTkButton(
    frame,
    text="Predict Image(s)",
    command=evaluate_train_model,
    corner_radius=10,
    hover_color="purple",
    font=("Arial Bold", 14),
    width=300,
)
predict_button.pack(pady=20)

documentation_text = ctk.CTkTextbox(
    frame,
    width=600,
    height=150,
    corner_radius=10,
    font=("Courier", 12),
    text_color="white",
)
documentation_text.pack(pady=10)
documentation_text.insert(
    "0.0",
    "This GUI is part of my final project\n"
    "in Computer Science in Machine Learning.\n\n"
    "Credits and Source Code:\n"
    "GitHub (source code)\n",
)

window.mainloop()
