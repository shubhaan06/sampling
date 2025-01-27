# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

# Load the dataset
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print("Dataset Preview:")
print(data.head())

# Check class distribution
print("\nClass Distribution:")
print(data['Class'].value_counts())

# Balance the dataset using a predefined technique
def balance_dataset(X, y, technique="smote"):
    if technique == "smote":
        sampler = SMOTE(random_state=42)
    elif technique == "oversampling":
        sampler = RandomOverSampler(random_state=42)
    elif technique == "undersampling":
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Unsupported balancing technique")
    
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    return X_resampled, y_resampled

# Predefine the balancing technique
balancing_technique = "smote"  # Change to "oversampling" or "undersampling" as needed
X = data.drop('Class', axis=1)  # Features
y = data['Class']              # Target
X_balanced, y_balanced = balance_dataset(X, y, technique=balancing_technique)

# Verify the new class distribution
print("\nBalanced Class Distribution:")
print(pd.Series(y_balanced).value_counts())

# Sample size calculation
def calculate_sample_size(N, e=0.05):
    return int(N / (1 + N * e**2))

# Sampling size
sample_size = calculate_sample_size(len(X_balanced))

# Define 5 sampling techniques
def random_sampling(X, y, size):
    return resample(X, y, n_samples=size, random_state=42)

def stratified_sampling(X, y, size):
    return resample(X, y, n_samples=size, stratify=y, random_state=42)

def cross_validation_sampling(X, y, size, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    sampled_indices = next(kf.split(X), None)[0]  # Get the first fold as the sample
    sampled_indices = sampled_indices[:size]  # Ensure the sample size
    return X.iloc[sampled_indices], y.iloc[sampled_indices]

def systematic_sampling(X, y, size):
    step = len(X) // size
    indices = np.arange(0, len(X), step)[:size]
    return X.iloc[indices], y.iloc[indices]

def bootstrap_sampling(X, y, size):
    return resample(X, y, n_samples=size, replace=True, random_state=42)

# Define 5 machine learning models
models = {
    "M1": RandomForestClassifier(random_state=42),
    "M2": LogisticRegression(max_iter=1000, random_state=42),
    "M3": SVC(random_state=42),
    "M4": KNeighborsClassifier(),
    "M5": GradientBoostingClassifier(random_state=42),
}

# Dictionary for sampling techniques
sampling_methods = {
    "Sampling1": random_sampling,
    "Sampling2": stratified_sampling,
    "Sampling3": cross_validation_sampling, 
    "Sampling4": systematic_sampling,
    "Sampling5": bootstrap_sampling,
}

# Evaluate each model with each sampling technique
results = {model: [] for model in models.keys()}

for sampling_name, sampling_func in sampling_methods.items():
    print(f"\nApplying {sampling_name}...")
    for model_name, model in models.items():
        # Apply sampling
        X_sample, y_sample = sampling_func(X_balanced, y_balanced, sample_size)
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[model_name].append(acc)
        print(f"{model_name} Accuracy ({sampling_name}): {acc:.2f}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results, index=sampling_methods.keys())
print("\nResults Table:")
print(results_df)

# Find the best sampling technique for each model
best_sampling = results_df.idxmax(axis=0)
print("\nBest Sampling Techniques for Each Model:")
print(best_sampling)

# Save results to a CSV file
results_df.to_csv("result_matrix.csv", index=True)
print("\nResults saved to 'sampling_results.csv'.")
