import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Define the main GUI application class
class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Prediction App")
        
        # Variables
        self.file_path = tk.StringVar()
        self.test_size = tk.DoubleVar(value=0.2)
        self.model_type = tk.StringVar(value="Linear Regression")
        
        # Create GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # File selection
        tk.Label(self.root, text="Excel File:").grid(row=0, column=0, sticky="e")
        tk.Entry(self.root, textvariable=self.file_path, width=50).grid(row=0, column=1)
        tk.Button(self.root, text="Browse", command=self.load_file).grid(row=0, column=2)
        
        # Model selection
        tk.Label(self.root, text="Select Model:").grid(row=1, column=0, sticky="e")
        tk.OptionMenu(self.root, self.model_type, "Linear Regression", "Random Forest").grid(row=1, column=1, sticky="w")
        
        # Test size
        tk.Label(self.root, text="Test Size (0-1):").grid(row=2, column=0, sticky="e")
        tk.Entry(self.root, textvariable=self.test_size, width=10).grid(row=2, column=1, sticky="w")
        
        # Start Button
        tk.Button(self.root, text="Run Prediction", command=self.run_prediction).grid(row=3, column=1, pady=10)

    def load_file(self):
        # File dialog to select Excel file
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx;*.xls")])
        self.file_path.set(file_path)

    def run_prediction(self):
        try:
            # Load the Excel file
            df = pd.read_excel(self.file_path.get())
            
            # Get columns for features and target
            columns = df.columns.tolist()
            feature_columns = [col for col in columns if col != "target"]  # Customize for your file
            target_column = "target"  # Customize for your file
            
            # Data Preprocessing
            X = df[feature_columns]
            y = df[target_column]
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size.get(), random_state=42
            )

            # Model Selection
            if self.model_type.get() == "Linear Regression":
                model = LinearRegression()
            elif self.model_type.get() == "Random Forest":
                model = RandomForestRegressor(random_state=42)
            else:
                messagebox.showerror("Error", "Unknown model selected.")
                return

            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)
            messagebox.showinfo("Model Performance", f"Mean Squared Error: {mse:.2f}")

            # Plot actual vs predicted values
            self.plot_results(y_test, y_pred)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def plot_results(self, y_test, y_pred):
        # Plotting results
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predicted vs Actual')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid()
        plt.show()


# Run the Application
root = tk.Tk()
app = MLApp(root)
root.mainloop()
