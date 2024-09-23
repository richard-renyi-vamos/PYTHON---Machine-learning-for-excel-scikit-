import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog, messagebox

# Load and preprocess the data
def load_data():
    file_path = filedialog.askopenfilename(title="Select Excel File", filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        data = pd.read_excel(file_path)
        messagebox.showinfo("File Loaded", f"Data loaded successfully from {file_path}")
        return data
    else:
        messagebox.showerror("Error", "File not selected!")
        return None

# Train the model
def train_model(data):
    # Assuming the last column is the target and all others are features
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using a Random Forest Classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Test accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    messagebox.showinfo("Model Trained", f"Model trained successfully with accuracy: {accuracy * 100:.2f}%")
    
    return model

# GUI for prediction
def predict(model, entries):
    input_data = []
    for entry in entries:
        input_data.append(float(entry.get()))
    
    input_data = [input_data]  # Convert to 2D array for the model
    prediction = model.predict(input_data)
    messagebox.showinfo("Prediction", f"The predicted class is: {prediction[0]}")

# GUI setup
def create_gui():
    window = tk.Tk()
    window.title("Machine Learning Predictor")
    
    # Load dataset button
    load_button = tk.Button(window, text="Load Dataset", command=lambda: load_data())
    load_button.grid(row=0, column=0, padx=10, pady=10)
    
    # Train model button
    train_button = tk.Button(window, text="Train Model", command=lambda: train_model(load_data()))
    train_button.grid(row=0, column=1, padx=10, pady=10)
    
    # Entry fields for user input
    labels = []
    entries = []
    
    for i in range(4):  # Assuming 4 input features (modify based on your dataset)
        label = tk.Label(window, text=f"Feature {i+1}:")
        label.grid(row=i+1, column=0, padx=10, pady=5)
        entry = tk.Entry(window)
        entry.grid(row=i+1, column=1, padx=10, pady=5)
        labels.append(label)
        entries.append(entry)
    
    # Predict button
    predict_button = tk.Button(window, text="Predict", command=lambda: predict(model, entries))
    predict_button.grid(row=6, column=1, padx=10, pady=10)
    
    window.mainloop()

if __name__ == "__main__":
    model = None  # Global variable to store the trained model
    create_gui()
