import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
from Data_Manipulation import *


def select_and_split() -> None:
    global output_path
    output_path = filedialog.askdirectory()
    path_entry.delete(0, "end")
    path_entry.insert(0, output_path)
    split_data(output_path)


def evaluate_model() -> None:
    global output_path
    classes = f"{Path(output_path).parent}/classes"
    output_path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras")])
    model_entry.delete(0, "end")
    model_entry.insert(0, output_path)
    preprocess_data(classes, output_path)


def open_github() -> None:
    webbrowser.open("https://github.com/Noamhotcoffee11/ml-project")


def open_pdf() -> None: ...

window = ctk.CTk()
window.title("Noam's project gui")
window.geometry("862x519")
window.resizable(False, False)

frame = ctk.CTkFrame(window, corner_radius=15)
frame.pack(fill="both", expand=True, padx=20, pady=20)

title = ctk.CTkLabel(frame, text="Welcome", font=("Arial Bold", 20))
title.pack(pady=10)

github_label = ctk.CTkButton(frame, text="Github", cursor="hand2", command=open_github, hover_color="purple")
github_label.pack(side="left", padx=50)

pdf_label = ctk.CTkButton(frame, text="pdf_book", cursor="hand2", command=open_pdf, hover_color="purple")
pdf_label.pack(side="right", padx=50)

path_entry = ctk.CTkEntry(frame, placeholder_text="Load the and split the data", corner_radius=10)
path_entry.pack(pady=20, fill="x", padx=20)

model_entry = ctk.CTkEntry(frame, placeholder_text="model's file", corner_radius=10)
model_entry.pack(pady=20, fill="x", padx=20)

path_button = ctk.CTkButton(
    frame, text="Select the data's path", command=select_and_split, hover_color="purple", corner_radius=10
)
path_button.pack(pady=10)

model_button = ctk.CTkButton(
    frame, text="Evaluate model", hover_color="purple", command=evaluate_model, corner_radius=10
)
model_button.pack(pady=10)

documention_text = ctk.CTkTextbox(frame, width=300, height=175, corner_radius=10)
documention_text.pack()
documention_text.insert(
    "0.0", "My GUI for the final project in Computer Science, Machine Learning,\n"
)
documention_text.insert("end", "\n")
documention_text.insert(
    "end", "Attributes for the book and GitHub page for the source code below:\n"
)
documention_text.insert("end", "\n")
documention_text.insert("end", "GitHub (source code)\n")
documention_text.insert("end", "\n")
documention_text.insert("end", "pdf-book (documentation)\n")

window.mainloop()
