import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold,RepeatedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Read the CSV file
file_path = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
df_pd = pd.read_csv(file_path, encoding='utf-8')

X = df_pd['processed'].values
y = df_pd['label'].values

kf = KFold(n_splits=2, shuffle=True, random_state=42)

# Iterate over the splits
i = 1
for train_index, test_index in kf.split(X):
    print(f"Fold {i}")
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    i += 1
print("shape x_train :", X_train.shape)
print("shape x_test :", X_test.shape)


def create_bag_of_words(X):
    print('Creating bag of words...')
    # Initialize the "TfidfVectorizer" object instead of "CountVectorizer"
    vectorizer = TfidfVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 ngram_range=(1, 2),
                                 max_features=10000
                                 )

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)

    # Convert to a NumPy array for easy handling
    train_data_features = train_data_features.toarray()

    # tfidf transform
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names_out()

    return vectorizer, vocab, train_data_features, tfidf_features, tfidf


vectorizer, vocab, train_data_features, tfidf_features, tfidf = \
    create_bag_of_words(X_train)

bag_dictionary = pd.DataFrame()
bag_dictionary['ngram'] = vocab
bag_dictionary['count'] = train_data_features[0]
bag_dictionary['tfidf_features'] = tfidf_features[0]

# Sort by raw count
bag_dictionary.sort_values(by=['count'], ascending=False, inplace=True)

# Initialize SVM pipeline
svm = Pipeline([('vect', TfidfVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                                         stop_words=None, ngram_range=(1, 2), max_features=10000)),
                ('tfidf', TfidfTransformer()),
                ('clf', SVC(C=1.0, kernel='poly', degree=2, gamma='auto')),
                ])
# Fit SVM pipeline
svm.fit(X_train, y_train)

# Predict using SVM
y_pred = svm.predict(X_test)

# Evaluate SVM
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred))

# Print Classification Report
print(classification_report(y_test, y_pred, zero_division=0))

# Print Accuracy
print('SVM Accuracy : ', accuracy_score(y_test, y_pred) * 100)


def predict_tweet_sentiment(tweet, model):
    tweet_processed = [tweet]  # Model expects a list of strings
    prediction = model.predict(tweet_processed)
    sentiment = prediction[0]
    if sentiment == 1:
        col = 'Positif'
    elif sentiment == 0:
        col = 'Negatif'
    else:
        col = 'Netral'  # Handle neutral or other sentiment values
    return col

# Example usage
user_tweet = input("Enter a tweet to predict its sentiment: ")
predicted_sentiment = predict_tweet_sentiment(user_tweet, svm)
print(f"The predicted sentiment of the tweet is: {predicted_sentiment}")