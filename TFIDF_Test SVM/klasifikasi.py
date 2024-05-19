import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load the dataset
file_path = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
df = pd.read_csv(file_path, encoding='utf-8')
df.columns = ['position', 'username', 'user_id', 'title', 'text', 'highlighs', 'link', 'location', 'sentimen', 'label', 'address', 'latitude', 'longitude', 'words', 'stop_words', 'stemmed_words', 'processed']

# Randomly shuffle the data
data = df.reindex(np.random.permutation(df.index))

# Extract features and target
X = data['processed'].values
y = data['sentimen'].values.astype(str)

# Define the pipeline
pipeline = Pipeline([
    ('count_vectorizer', CountVectorizer(ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer()),
    ('classifier', OneVsRestClassifier(LinearSVC()))
])

# Define KFold cross-validation
k_fold = KFold(n_splits=6, shuffle=True, random_state=42)

# Initialize variables for metrics
overall_accuracy = []

# Perform cross-validation
for fold, (train_indices, test_indices) in enumerate(k_fold.split(X)):
    print(f"Fold {fold+1}")

    # Split data into train and test sets
    train_text = X[train_indices]
    train_y = y[train_indices]

    test_text = X[test_indices]
    test_y = y[test_indices]

    # Fit pipeline on training data
    pipeline.fit(train_text, train_y)

    # Predict on test data
    predictions = pipeline.predict(test_text)

    # Calculate accuracy
    accuracy = accuracy_score(test_y, predictions)
    overall_accuracy.append(accuracy)

    # Print results
    print(f"Accuracy: {accuracy}")
    print(classification_report(test_y, predictions, digits=2, zero_division=0))
    print(confusion_matrix(test_y, predictions, labels=np.unique(y)))

    print("\n")

# Calculate and print overall accuracy
overall_accuracy = np.mean(overall_accuracy)
print(f"Overall Accuracy: {overall_accuracy * 100:.2f}%")

# Calculate confusion matrix percentages
cm = confusion_matrix(test_y, predictions, labels=np.unique(y))
percentage_matrix = 100 * cm / cm.sum(axis=1)[:, np.newaxis]

# Handle NaN values in percentage matrix
percentage_matrix = np.nan_to_num(percentage_matrix, nan=0.0)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix (Percentage)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
