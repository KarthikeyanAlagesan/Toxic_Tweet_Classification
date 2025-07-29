# main.py
import pandas as pd

# Load the CSV file
df = pd.read_csv(r"C:\Users\karth\Documents\Toxic_Tweet\FinalBalancedDataset.csv")

# Display first 5 rows
print("Sample data:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
import re

def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z\s]", "", text)  # remove special characters
    text = text.lower()
    return text

df['cleaned_text'] = df['tweet'].apply(clean_text)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

X = df['cleaned_text']
y = df['Toxicity']  # 1 = toxic, 0 = non-toxic

# Try Bag of Words first
vectorizer = CountVectorizer()
X_features = vectorizer.fit_transform(X)

# For TF-IDF instead, replace with:
# vectorizer = TfidfVectorizer()
# X_features = vectorizer.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "K-NN": KNeighborsClassifier(),
    "SVM": SVC(probability=True)
}

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
