import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Simple Tkinter Example")
root.geometry("300x200")  # Set the window size

# Create a label widget
label = tk.Label(root, text="Hello, Tkinter!", font=("Arial", 14))
label.pack(pady=50)  # Add some padding for aesthetics

# Run the application
root.mainloop()
