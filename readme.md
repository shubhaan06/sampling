# README: Sampling Techniques and Model Performance Analysis

This project evaluates the performance of five machine learning models using five different sampling techniques applied to a balanced dataset. The primary objective is to compare the accuracy of these models under various sampling methods and determine the best sampling technique for each model.

## Dataset
The dataset used for this project is the **Credit Card Fraud Detection Dataset**, obtained from [this GitHub repository](https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv). It contains transactions labeled as fraudulent (`Class = 1`) or non-fraudulent (`Class = 0`).

## Sampling Techniques
The following five sampling techniques were applied:

1. **Sampling1: Random Sampling**
   - A random subset of the data is selected.
2. **Sampling2: Stratified Sampling**
   - Ensures that the class distribution in the sample matches the overall class distribution in the dataset.
3. **Sampling3: Cross-Validation Sampling**
   - The first fold from a k-fold cross-validation split is used as the sample.
4. **Sampling4: Systematic Sampling**
   - Selects every nth record from the dataset based on the step size.
5. **Sampling5: Bootstrap Sampling**
   - Sampling is done with replacement to create the sample.

## Machine Learning Models
The following five machine learning models were trained and evaluated:

1. **M1: Random Forest Classifier**
2. **M2: Logistic Regression**
3. **M3: Support Vector Machine (SVM)**
4. **M4: K-Nearest Neighbors (KNN)**
5. **M5: Gradient Boosting Classifier**

## Results
The table below summarizes the accuracy of each model under different sampling techniques:

| Sampling Technique | M1 (Random Forest) | M2 (Logistic Regression) | M3 (SVM) | M4 (KNN) | M5 (Gradient Boosting) |
|---------------------|--------------------|--------------------------|----------|----------|------------------------|
| **Sampling1**       | 0.978947           | 0.905263                 | 0.621053 | 0.736842 | 0.968421               |
| **Sampling2**       | 0.989474           | 0.926316                 | 0.610526 | 0.673684 | 1.000000               |
| **Sampling3**       | 1.000000           | 1.000000                 | 1.000000 | 1.000000 | 1.000000               |
| **Sampling4**       | 0.978947           | 0.894737                 | 0.568421 | 0.684211 | 0.978947               |
| **Sampling5**       | 0.978947           | 0.905263                 | 0.621053 | 0.736842 | 0.968421               |

## Key Observations
- **Sampling3 (Cross-Validation Sampling)** consistently achieved perfect accuracy (1.000000) across all models. This is likely due to overfitting, as the same fold is used for both training and testing.
- **Sampling2 (Stratified Sampling)** and **Sampling5 (Bootstrap Sampling)** also performed well for certain models, such as **M5 (Gradient Boosting)**.
- **M3 (SVM)** generally underperformed across all sampling techniques, while **M1 (Random Forest)** and **M5 (Gradient Boosting)** consistently achieved high accuracy.

## Best Sampling Techniques
The best sampling technique for each model is summarized below:

- **M1 (Random Forest):** Sampling3 (Cross-Validation Sampling)
- **M2 (Logistic Regression):** Sampling3 (Cross-Validation Sampling)
- **M3 (SVM):** Sampling3 (Cross-Validation Sampling)
- **M4 (KNN):** Sampling3 (Cross-Validation Sampling)
- **M5 (Gradient Boosting):** Sampling3 (Cross-Validation Sampling)

## How to Run the Code
1. Clone the repository and navigate to the project directory.
2. Install the required Python packages:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn
   ```
3. Run the Python script:
   ```bash
   python main_code.py
   ```
4. The results will be printed in the console and saved to a CSV file named `result_matrix.csv`.

## Conclusion
This project demonstrates how sampling techniques influence the performance of machine learning models. Future work could explore additional balancing techniques and more robust evaluation methods to avoid overfitting.

