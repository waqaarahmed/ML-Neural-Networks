# Document Intelligence Model

This Python script implements a document intelligence model for text classification using machine learning techniques. The model is trained to classify documents into predefined categories based on their content.

## Requirements

- Python 3.12
- Libraries: `pandas`, `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas scikit-learn
```
## Project Structure

- `main.py`: Python script containing the document intelligence model implementation.

## Usage

1. Ensure you have the necessary dependencies installed.
2. Place your annotated data in the `annotated_data` directory. The data should be in CSV format with columns for text content (`text`) and corresponding labels (`label`).
3. Run the Python script `document_intelligence_model.py`.
4. The script will perform the following steps:
5. Step 1: Load annotated data: Load the annotated data from the CSV file.
6. Step 2: Preprocess data: Preprocess the data by extracting text content and corresponding labels.
7. Step 3: Feature extraction: Extract features from the text data using TF-IDF vectorization.
8. Step 4: Train-test split: Split the dataset into training and testing sets.
9. Step 5: Model training: Train a logistic regression model on the training data.
10. Step 6: Model evaluation: Evaluate the trained model's performance on the testing data by calculating accuracy and generating a classification report.
11. Step 7: Save the trained model: Save the trained model to a file using joblib.

## Note

1. The document intelligence model uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical features.
2. Adjust the parameters of TF-IDF vectorization and the machine learning model as needed to improve performance.
3. Experiment with different preprocessing techniques and machine learning algorithms to optimize the model for your specific use case.
4. For larger datasets or more complex classification tasks, consider using more advanced machine learning algorithms or deep learning techniques.
