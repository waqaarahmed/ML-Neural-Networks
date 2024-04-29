import os
import glob
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Loading annotated data
data_path = "annotated_data/"
annotations = pd.read_csv(os.path.join(data_path, "annotations.csv"))

#Preprocessing data
#For simplicity, let's assume the PDF text has been extracted and stored in 'text' column
X = annotations['text']
y = annotations['label']

#Feature extraction
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

#Model training
model = LogisticRegression()
model.fit(X_train, y_train)

#Evaluating Model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Saving the trained model
model_output_path = "trained_models/"
if not os.path.exists(model_output_path):
    os.makedirs(model_output_path)
model_output_file = os.path.join(model_output_path, "text_classifier.pkl")
joblib.dump(model, model_output_file)
