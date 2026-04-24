import numpy as np
import pandas as pd
import os
from google import genai
from dotenv import load_dotenv
import time


class PFNWrapper:
    def __init__(self, reg, prob_adj=None):
        self.reg = reg
        self.prob_adj = prob_adj
    
    def fit(self, X_train, y_train):
        self.numeric_cols = X_train.select_dtypes(include=['number']).columns
        self.train_mean_X = X_train[self.numeric_cols].mean()
        self.train_std_X = X_train[self.numeric_cols].std()
        # clip std to avoid division by zero
        self.train_std_X = np.maximum(self.train_std_X, 1.0)
        X_train_scaled = X_train.copy()
        X_train_scaled[self.numeric_cols] = (X_train[self.numeric_cols] - self.train_mean_X) / self.train_std_X

        self.train_mean_y = y_train.mean()
        self.train_std_y = y_train.std()
        # clip std to avoid division by zero
        self.train_std_y = np.maximum(self.train_std_y, 1.0)
        y_train_scaled = (y_train - self.train_mean_y) / self.train_std_y
        
        self.reg.fit(X_train_scaled.to_numpy(), y_train_scaled.to_numpy())
        
    def predict(self, X_test):
        X_test_scaled = X_test.copy()
        X_test_scaled[self.numeric_cols] = (X_test[self.numeric_cols] - self.train_mean_X) / self.train_std_X
        
        if self.prob_adj is None:
            preds_scaled = self.reg.predict(X_test_scaled.to_numpy())
        else:
            preds_scaled = self.reg.predict(X_test_scaled.to_numpy(), prob_adj=self.prob_adj)
        preds_final = (preds_scaled * self.train_std_y) + self.train_mean_y
        
        return preds_final
    
    
class LLMRegressor:
    def __init__(self, prob_adj=None, model='gemini-2.5-flash-lite'):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))
        self.model = model
        self.prob_adj = prob_adj
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.node_names = X_train.columns.tolist() + [y_train.name]
        if self.prob_adj is not None:
            # construct causal summary
            lines = []
            n = len(self.node_names)
            for i in range(n - 1):
                feature_name = self.node_names[i]
                # Feature i -> Target
                prob_to_target = self.prob_adj[i, n - 1].item()
                lines.append(f"There is a causal influence of {feature_name} on {y_train.name} with probability {prob_to_target:.2f}")
                # Target -> Feature i
                prob_from_target = self.prob_adj[n - 1, i].item()
                lines.append(f"There is a causal influence of {y_train.name} on {feature_name} with probability {prob_from_target:.2f}")

            self.causal_summary = "\n".join(lines) + "\n"
        
    def serialize(self, X_train, y_train, X_test):
        train_df = pd.concat([X_train, y_train], axis=1)
        train_csv = train_df.to_csv(index=False)
        test_csv = X_test.to_csv(index=False)
        target_name = y_train.name
        
        lines = []
        lines.append(f"You are a regression model. Your task is to predict the value of '{target_name}' for the provided test sample based on the patterns found in the training data.")
        if self.prob_adj is not None:
            lines.append("You are also given a number of causal relationships between the features and the target, which may help you make a better prediction.")
        lines.append("TRAINING DATA")
        lines.append(f"Below are examples of inputs and their corresponding {target_name}:")
        lines.append(train_csv)
        if self.prob_adj is not None:
            lines.append("CAUSAL INFORMATION")
            lines.append(self.causal_summary)
        lines.append("TEST DATA")
        lines.append(f"Predict the '{target_name}' for the following row:")
        lines.append(test_csv)
        lines.append("INSTRUCTIONS")
        lines.append("1. Analyze the relationship between features and the target.")
        lines.append("2. Provide only the predicted numeric value for the test data.")
        lines.append("3. Return a single float number without any additional text or formatting.")
        lines.append("4. Do not include any explanations or prose.")
        lines.append("Prediction:")
        
        return "\n".join(lines)
        
    def predict(self, X_test):
        preds = []
        for i in range(len(X_test)):
            row = X_test[i:i+1]
            prompt = self.serialize(self.X_train, self.y_train, row)
            print(prompt)
            start_time = time.time()
            response = self.client.models.generate_content(model=self.model, contents=prompt).text.strip()
            end_time = time.time()
            print(f"LLM response: {response} (took {end_time - start_time:.2f} seconds)")
            preds.append(float(response))
        return np.array(preds)