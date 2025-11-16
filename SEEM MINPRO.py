import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import arff

# Dictionary to store user credentials for simplicity (in production, use a proper database or file)
user_credentials = {"admin": "password123"}

# LoginWindow Class - Handles login form
class LoginWindow:
    def __init__(self, root, proceed_callback):
        self.root = root
        self.proceed_callback = proceed_callback
        self.root.title("Login")
        self.root.geometry("300x250")
        
        self.label_userid = tk.Label(self.root, text="User ID:")
        self.label_userid.pack(pady=10)
        
        self.entry_userid = tk.Entry(self.root)
        self.entry_userid.pack(pady=10)
        
        self.label_password = tk.Label(self.root, text="Password:")
        self.label_password.pack(pady=10)
        
        self.entry_password = tk.Entry(self.root, show="*")
        self.entry_password.pack(pady=10)
        
        self.login_button = tk.Button(self.root, text="Login", command=self.validate_credentials)
        self.login_button.pack(pady=10)

        self.create_account_button = tk.Button(self.root, text="Create Account", command=self.open_create_account_window)
        self.create_account_button.pack(pady=10)
        
    def validate_credentials(self):
        entered_userid = self.entry_userid.get()
        entered_password = self.entry_password.get()
        
        if entered_userid in user_credentials and user_credentials[entered_userid] == entered_password:
            self.proceed_callback()  # Proceed to the main model interface
            self.root.destroy()  # Close the login window
        else:
            messagebox.showerror("Error", "Invalid User ID or Password.")
    
    def open_create_account_window(self):
        self.create_account_window = tk.Toplevel(self.root)
        CreateAccountWindow(self.create_account_window)


# CreateAccountWindow Class - Handles account creation
class CreateAccountWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Create Account")
        self.root.geometry("300x250")

        self.label_userid = tk.Label(self.root, text="User ID:")
        self.label_userid.pack(pady=10)

        self.entry_userid = tk.Entry(self.root)
        self.entry_userid.pack(pady=10)

        self.label_password = tk.Label(self.root, text="Password:")
        self.label_password.pack(pady=10)

        self.entry_password = tk.Entry(self.root, show="*")
        self.entry_password.pack(pady=10)

        self.label_confirm_password = tk.Label(self.root, text="Confirm Password:")
        self.label_confirm_password.pack(pady=10)

        self.entry_confirm_password = tk.Entry(self.root, show="*")
        self.entry_confirm_password.pack(pady=10)

        self.create_button = tk.Button(self.root, text="Create Account", command=self.create_account)
        self.create_button.pack(pady=20)

    def create_account(self):
        username = self.entry_userid.get()
        password = self.entry_password.get()
        confirm_password = self.entry_confirm_password.get()

        # Validation
        if username in user_credentials:
            messagebox.showerror("Error", "Username already exists.")
        elif password != confirm_password:
            messagebox.showerror("Error", "Passwords do not match.")
        elif not username or not password:
            messagebox.showerror("Error", "Username and Password cannot be empty.")
        else:
            # Save the new account (for simplicity, stored in a dictionary here)
            user_credentials[username] = password
            messagebox.showinfo("Success", "Account created successfully!")
            self.root.destroy()


# ModelInterface Class - Main model prediction and evaluation logic
class ModelInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Prediction Model")
        self.root.geometry("800x600")
        
        # Add buttons and labels for user interactions
        self.upload_button = tk.Button(self.root, text="Upload ARFF File", command=self.load_arff_file)
        self.upload_button.pack(pady=20)
        
        self.model_button = tk.Button(self.root, text="Run Models", state=tk.DISABLED, command=self.run_models)
        self.model_button.pack(pady=20)
        
        self.results_label = tk.Label(self.root, text="", justify=tk.LEFT)
        self.results_label.pack(pady=20)
        
        self.filepath = None
        self.df = None
        self.X = None
        self.y_effort = None
        self.y_duration = None
        self.y_cost = None

    def load_arff_file(self):
        # File dialog to upload the ARFF file
        self.filepath = filedialog.askopenfilename(filetypes=[("ARFF files", "*.arff")])
        if not self.filepath:
            return
        
        try:
            # Load ARFF file and convert it to pandas DataFrame
            data = arff.load(open(self.filepath, 'r'))
            self.df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

            # Feature selection
            self.X = self.df[['LOC', 'Complexity', 'DomainKnowledge', 'TeamExperience']]
            self.y_effort = self.df['Effort']
            self.y_duration = self.df['Duration']
            self.y_cost = self.df['ProjectCost']
            
            self.model_button.config(state=tk.NORMAL)
            self.results_label.config(text="Data loaded successfully! Ready to run models.")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")

    def run_models(self):
        if self.df is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        
        # Train-test split
        X_train, X_test, y_train_effort, y_test_effort = train_test_split(self.X, self.y_effort, test_size=0.2, random_state=42)
        _, _, y_train_duration, y_test_duration = train_test_split(self.X, self.y_duration, test_size=0.2, random_state=42)
        _, _, y_train_cost, y_test_cost = train_test_split(self.X, self.y_cost, test_size=0.2, random_state=42)

        # Linear Regression Model
        model_lr_effort = LinearRegression()
        model_lr_effort.fit(X_train, y_train_effort)
        y_pred_lr_effort = model_lr_effort.predict(X_test)

        # KNN Model
        knn_model_effort = KNeighborsRegressor(n_neighbors=5)
        knn_model_effort.fit(X_train, y_train_effort)
        y_pred_knn_effort = knn_model_effort.predict(X_test)

        # Evaluation Metrics
        mse_lr_effort = mean_squared_error(y_test_effort, y_pred_lr_effort)
        r2_lr_effort = r2_score(y_test_effort, y_pred_lr_effort)

        mse_knn_effort = mean_squared_error(y_test_effort, y_pred_knn_effort)
        r2_knn_effort = r2_score(y_test_effort, y_pred_knn_effort)

        results_text = f"Effort Model Results:\n"
        results_text += f"Linear Regression MSE: {mse_lr_effort:.2f}, R-squared: {r2_lr_effort:.2f}\n"
        results_text += f"KNN MSE: {mse_knn_effort:.2f}, R-squared: {r2_knn_effort:.2f}\n"

        # Plotting the results
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test_effort, y_pred_lr_effort, color='blue', alpha=0.6, edgecolors='w', s=100)
        plt.plot([self.y_effort.min(), self.y_effort.max()], [self.y_effort.min(), self.y_effort.max()], 'k--', lw=4)
        plt.xlabel('Actual Effort')
        plt.ylabel('Predicted Effort')
        plt.title('Actual vs Predicted (Linear Regression)')

        plt.subplot(1, 2, 2)
        plt.scatter(y_test_effort, y_pred_knn_effort, color='red', alpha=0.6, edgecolors='w', s=100)
        plt.plot([self.y_effort.min(), self.y_effort.max()], [self.y_effort.min(), self.y_effort.max()], 'k--', lw=4)
        plt.xlabel('Actual Effort')
        plt.ylabel('Predicted Effort')
        plt.title('Actual vs Predicted (KNN)')

        plt.tight_layout()
        plt.show()

        # Display the best model
        if r2_lr_effort > r2_knn_effort:
            results_text += f"Best model for Effort: Linear Regression with R-squared: {r2_lr_effort:.2f}\n"
        else:
            results_text += f"Best model for Effort: KNN with R-squared: {r2_knn_effort:.2f}\n"

        self.results_label.config(text=results_text)

# Creating the Tkinter root window
root = tk.Tk()

# Create and show the login window
login_window = LoginWindow(root, lambda: ModelInterface(tk.Tk()))  # Pass a callback to open the main interface
root.mainloop()
 