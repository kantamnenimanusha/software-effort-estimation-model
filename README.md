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

# Simple in-memory user authentication
user_credentials = {"admin": "password123"}

# ------------------------------------------
# LOGIN WINDOW
# ------------------------------------------
class LoginWindow:
    def __init__(self, root, proceed_callback):
        self.root = root
        self.proceed_callback = proceed_callback
        self.root.title("Login")
        self.root.geometry("300x250")

        tk.Label(self.root, text="User ID:").pack(pady=10)
        self.entry_userid = tk.Entry(self.root)
        self.entry_userid.pack(pady=10)

        tk.Label(self.root, text="Password:").pack(pady=10)
        self.entry_password = tk.Entry(self.root, show="*")
        self.entry_password.pack(pady=10)

        tk.Button(self.root, text="Login", command=self.validate_credentials).pack(pady=10)
        tk.Button(self.root, text="Create Account", command=self.open_create_account_window).pack(pady=10)

    def validate_credentials(self):
        user = self.entry_userid.get()
        password = self.entry_password.get()
        if user in user_credentials and user_credentials[user] == password:
            self.proceed_callback()
            self.root.destroy()
        else:
            messagebox.showerror("Error", "Invalid User ID or Password.")

    def open_create_account_window(self):
        CreateAccountWindow(tk.Toplevel(self.root))


# ------------------------------------------
# CREATE ACCOUNT WINDOW
# ------------------------------------------
class CreateAccountWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Create Account")
        self.root.geometry("300x250")

        tk.Label(self.root, text="User ID:").pack(pady=10)
        self.entry_userid = tk.Entry(self.root)
        self.entry_userid.pack(pady=10)

        tk.Label(self.root, text="Password:").pack(pady=10)
        self.entry_password = tk.Entry(self.root, show="*")
        self.entry_password.pack(pady=10)

        tk.Label(self.root, text="Confirm Password:").pack(pady=10)
        self.entry_confirm = tk.Entry(self.root, show="*")
        self.entry_confirm.pack(pady=10)

        tk.Button(self.root, text="Create Account", command=self.create_account).pack(pady=20)

    def create_account(self):
        user = self.entry_userid.get()
        password = self.entry_password.get()
        confirm = self.entry_confirm.get()

        if user in user_credentials:
            messagebox.showerror("Error", "Username already exists.")
        elif password != confirm:
            messagebox.showerror("Error", "Passwords do not match.")
        elif not user or not password:
            messagebox.showerror("Error", "Fields cannot be empty.")
        else:
            user_credentials[user] = password
            messagebox.showinfo("Success", "Account created successfully!")
            self.root.destroy()


# ------------------------------------------
# MAIN MODEL INTERFACE
# ------------------------------------------
class ModelInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Software Effort Estimation")
        self.root.geometry("800x600")

        tk.Button(self.root, text="Upload ARFF File", command=self.load_arff_file).pack(pady=20)
        self.model_button = tk.Button(self.root, text="Run Models", state=tk.DISABLED, command=self.run_models)
        self.model_button.pack(pady=20)

        self.results_label = tk.Label(self.root, text="", justify=tk.LEFT)
        self.results_label.pack(pady=20)

        self.filepath = None
        self.df = None

    def load_arff_file(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("ARFF files", "*.arff")])
        if not self.filepath:
            return

        try:
            data = arff.load(open(self.filepath, 'r'))
            self.df = pd.DataFrame(data['data'], columns=[attr[0] for attr in data['attributes']])

            self.X = self.df[['LOC', 'Complexity', 'DomainKnowledge', 'TeamExperience']]
            self.y_effort = self.df['Effort']

            self.model_button.config(state=tk.NORMAL)
            self.results_label.config(text="Data loaded successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {e}")

    def run_models(self):
        if self.df is None:
            messagebox.showerror("Error", "No data loaded.")
            return

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_effort, test_size=0.2, random_state=42)

        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        # KNN Regression
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)

        mse_lr = mean_squared_error(y_test, y_pred_lr)
        mse_knn = mean_squared_error(y_test, y_pred_knn)

        r2_lr = r2_score(y_test, y_pred_lr)
        r2_knn = r2_score(y_test, y_pred_knn)

        result = (
            f"Effort Prediction Results:\n"
            f"Linear Regression -> MSE: {mse_lr:.2f}, R²: {r2_lr:.2f}\n"
            f"KNN Regression -> MSE: {mse_knn:.2f}, R²: {r2_knn:.2f}\n"
        )

        best = "Linear Regression" if r2_lr > r2_knn else "KNN Regression"
        result += f"\nBest Model: {best}"

        self.results_label.config(text=result)

        # Visualization
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred_lr)
        plt.title("Linear Regression")
        plt.xlabel("Actual Effort")
        plt.ylabel("Predicted Effort")

        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_knn)
        plt.title("KNN Regression")
        plt.xlabel("Actual Effort")
        plt.ylabel("Predicted Effort")

        plt.tight_layout()
        plt.show()


# ------------------------------------------
# RUN APPLICATION
# ------------------------------------------
root = tk.Tk()
LoginWindow(root, lambda: ModelInterface(tk.Tk()))
root.mainloop()
