# Toxic_Tweet_Classification
To build a machine learning model that can predict whether a tweet is toxic or non-toxic (binary classification) based on its text content.
| Tool/Library                                                  | Purpose                     |
| ------------------------------------------------------------- | --------------------------- |
| `pandas`                                                      | Data handling               |
| `scikit-learn`                                                | Machine learning algorithms |
| `matplotlib`                                                  | Plotting graphs             |
| `re`                                                          | Text cleaning with regex    |
| `TF-IDF`, `Bag of Words`                                      | Feature extraction          |
| `Decision Tree`, `Random Forest`, `Naive Bayes`, `SVM`, `KNN` | Classification models       |

# We import the pandas library, which is essential for data manipulation and analysis in Python.
import pandas as pd
#  Load the CSV file
df = pd.read_csv(r"C:\Users\karth\Documents\Toxic_Tweet\FinalBalancedDataset.csv")
# df.head() displays the first 5 rows of the dataset to quickly check the structure and content of the data.This helps verify if the file is loaded correctly
# Display first 5 rows
print("Sample data:")
print(df.head())
# This code checks for null or missing values in each column using df.isnull().sum().This step is important before any preprocessing or training, as missing data can lead to errors or skewed model performance.
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
# Imports the re module which provides support for regular expressions in Python. This is useful for text cleaning and pattern matching, especially in NLP tasks like removing URLs, punctuation, or special characters. 
import re

| Line                               | Explanation                                       |
| ---------------------------------- | ------------------------------------------------- |
| `def clean_text(text):`            | Defines a function to clean each tweet's text.    |
| `text = str(text)`                 | Converts input to a string (in case it's not).    |
| `re.sub(r"http\S+", "", text)`     | Removes URLs that start with "http".              |
| `re.sub(r"[^A-Za-z\s]", "", text)` | Removes all characters except letters and spaces. |
| `text = text.lower()`              | Converts all letters to lowercase.                |
| `return text`                      | Returns the cleaned text string.                  |

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove special characters
    text = text.lower()
    return text

✔ Applies the clean_text function to every row in the 'tweet' column and creates a new column called 'cleaned_text'
df['cleaned_text'] = df['tweet'].apply(clean_text)

✔ Imports two vectorizers:
CountVectorizer: Bag of Words (word counts)
TfidfVectorizer: Term Frequency-Inverse Document Frequency
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
| Variable | Meaning                                          |
| -------- | ------------------------------------------------ |
| `X`      | The input features: cleaned tweets               |
| `y`      | The labels (target variable): Toxic or Non-Toxic |
X = df['cleaned_text']
y = df['Toxicity']  # 1 = toxic, 0 = non-toxic

| Line                | Explanation                                                       |
| ------------------- | ----------------------------------------------------------------- |
| `CountVectorizer()` | Initializes a Bag-of-Words transformer                            |
| `.fit_transform(X)` | Learns vocab and converts each tweet into a vector of word counts |
vectorizer = CountVectorizer()
X_features = vectorizer.fit_transform(X)

# Optional TF-IDF Alternative:
# vectorizer = TfidfVectorizer()
# X_features = vectorizer.fit_transform(X)

✔ Splits the data into training (80%) and testing (20%) sets.
The random_state=42 ensures reproducibility.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

✔ Creates a dictionary of model names with their respective scikit-learn objects.
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "K-NN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

| Line                                  | Explanation                                |
| ------------------------------------- | ------------------------------------------ |
| `for name, model in models.items()`   | Loops through each model in the dictionary |
| `model.fit(X_train, y_train)`         | Trains the model on the training data      |
| `y_pred = model.predict(X_test)`      | Predicts toxicity on the test set          |
| `classification_report(...)`          | Prints precision, recall, F1-score         |
| `confusion_matrix(...)`               | Shows confusion matrix (TP, TN, FP, FN)    |
| `roc_auc_score(...)`                  | Computes ROC-AUC score                     |
| `RocCurveDisplay.from_estimator(...)` | Plots the ROC curve                        |
| `plt.show()`                          | Displays the curve for each model          |

for name, model in models.items():
    print(f"\n📌 Model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {name}")
    plt.show()

📌 Model: Random Forest
Classification Report:
               precision    recall  f1-score   support
           0       0.93      0.95      0.94       100
           1       0.94      0.92      0.93       100
Confusion Matrix:
 [[95  5]
  [ 8 92]]
ROC-AUC Score: 0.935






