import os
import csv
import pandas as pd
import hashlib
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
class UserDatabase:
    def __init__(self):
        # Create exercise folder if it doesn't exist
        if not os.path.exists("Exercise"):
            os.makedirs("Exercise")
        
        # Path to the user credentials file
        self.credentials_file = "Exercise/user_credentials.csv"
        
        # Initialize user credentials file if it doesn't exist
        if not os.path.exists(self.credentials_file):
            self._create_initial_users()
    
    def _create_initial_users(self):
        """Create 25 dummy users with credentials"""
        with open(self.credentials_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'password_hash'])
            
            for i in range(1, 26):
                username = f"AIBT{i}"
                password = f"AICCORE{i}"
                # Hash the password for security
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                writer.writerow([username, password_hash])
    
    def verify_user(self, username, password):
        """Verify if username and password are correct"""
        try:
            df = pd.read_csv(self.credentials_file)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            # Check if username exists and password matches
            user_row = df[df['username'] == username]
            if not user_row.empty and user_row.iloc[0]['password_hash'] == password_hash:
                return True
            return False
        except Exception as e:
            print(f"Error verifying user: {e}")
            return False
    
    def save_user_interaction(self, username, question, model_responses):
        """Save user interaction to a CSV file"""
        try:
            # Create a folder for the user if it doesn't exist
            user_folder = f"Exercise/{username}"
            if not os.path.exists(user_folder):
                os.makedirs(user_folder)
            
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_folder}/{username}_interaction_{timestamp}.csv"
            
            # Prepare data for saving
            data = {
                'question': [question],
                'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
            }
            
            # Add model responses
            for model, response in model_responses.items():
                data[f"model_{model}"] = [response]
            
            # Create dataframe and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            return filename
        except Exception as e:
            print(f"Error saving user interaction: {e}")
            return None
    
    def get_user_history(self, username):
        """Get all interactions for a specific user"""
        user_folder = f"Exercise/{username}"
        if not os.path.exists(user_folder):
            return []
        
        files = [f for f in os.listdir(user_folder) if f.endswith('.csv')]
        return files