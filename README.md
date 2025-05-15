
# K-Nearest Neighbors Classification for Gamma/Hadron Dataset

This repository contains a Jupyter notebook implementing **K-Nearest Neighbors (KNN)** classification on the MAGIC Gamma Telescope dataset (`magic04.data`). The project focuses on classifying gamma and hadron events, including data preprocessing, model training, hyperparameter tuning, and evaluation.

## Overview

The notebook (`classification_knn.ipynb`) implements:
- **Data Preprocessing**: Loads and balances the dataset, standardizes features.
- **KNN Classification**: Trains a KNN model and evaluates performance for different `k` values.
- **Hyperparameter Tuning**: Identifies the optimal `k` based on the F1-score.
- **Evaluation**: Assesses model performance on a test set using multiple metrics.
- **Result Analysis**: Provides insights into model performance and biases.

## Features

- **Dataset Balancing**: Matches gamma and hadron samples to address class imbalance.
- **Feature Standardization**: Uses `StandardScaler` to normalize features for improved KNN performance.
- **Hyperparameter Optimization**: Tests `k` values from 1 to 20 to find the best F1-score.
- **Comprehensive Evaluation**: Reports accuracy, precision, recall, F1-score, confusion matrix, MSE, and MAE.
- **Result Interpretation**: Analyzes model performance and potential biases in classification.


## Dataset

The dataset used is `magic04.data`, which contains 10 features describing gamma and hadron events, with a binary class label (`g` for gamma, `h` for hadron). The features are:

- `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`
- `class`: Binary label (gamma: 1, hadron: 0)

Place the dataset file (`magic04.data`) in the same directory as the notebook.

## Directory Structure

- `magic04.data`: Input dataset file.
- `classification_knn.ipynb`: Main Jupyter notebook.
- `README.md`: This file.

## Usage

1. **Prepare the Dataset**:
   - Ensure `magic04.data` is in the same directory as the notebook.
   - The dataset is loaded and preprocessed automatically (balancing and standardization).

2. **Run the Notebook**:
   - Open `classification_knn.ipynb` in Jupyter.
   - Execute cells sequentially to:
     - Load and preprocess the dataset.
     - Split data into training (70%), validation (15%), and test (15%) sets.
     - Evaluate KNN for `k` values from 1 to 20 on the validation set.
     - Train the optimal KNN model (best `k` based on F1-score).
     - Evaluate the model on the test set and display metrics.

3. **Key Steps**:
   - **Balancing**: The dataset is balanced by sampling equal numbers of gamma and hadron events.
   - **Standardization**: Features are standardized using `StandardScaler`.
   - **Hyperparameter Tuning**: The optimal `k` (e.g., 17) is selected based on the highest F1-score on the validation set.
   - **Evaluation**: Metrics include accuracy (~80.4%), precision (~75.4%), recall (~88.6%), F1-score (~81.5%), confusion matrix, MSE, and MAE.

## Results

The notebook outputs the following for the test set (with optimal `k=17`):
- **Accuracy**: ~80.4% (80% of predictions are correct).
- **Precision**: ~75.4% (75.4% of predicted gamma events are correct).
- **Recall**: ~88.6% (88.6% of actual gamma events are detected).
- **F1-Score**: ~81.5% (balanced precision and recall).
- **Confusion Matrix**: Indicates 282 false positives and 111 false negatives, suggesting a slight bias toward gamma classification.
- **MSE/MAE**: ~0.196, indicating low prediction errors.

## Notes

- **Class Imbalance**: The dataset is balanced to prevent bias toward the majority class (gamma).
- **Feature Scaling**: Standardization is critical for KNN, as it relies on distance metrics.
- **Deprecation Warning**: A `FutureWarning` may appear for `pandas.replace` due to downcasting behavior. This can be ignored or addressed by setting `pd.set_option('future.no_silent_downcasting', True)`.
- **Performance**: The model performs well, with high recall for gamma events, but precision could be improved to reduce false positives.

## Future Improvements

- Explore other distance metrics (e.g., Manhattan, weighted KNN).
- Implement cross-validation for more robust `k` selection.
- Address false positives by adjusting class weights or using ensemble methods.
- Visualize feature importance or decision boundaries for better interpretability.


For issues or contributions, please contact me basemhesham200318@gmail.com or open a pull request or issue on this GitHub repository.
