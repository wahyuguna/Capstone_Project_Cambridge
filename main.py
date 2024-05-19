from flask import Flask, render_template, url_for, request, abort, make_response
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import random
import os
import shutil


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file_path = r"C:\Users\USER\PycharmProjects\Capstone_Project_Cambridge\dataset\hasil_preprocessing_data.csv"
    df_pd = pd.read_csv(file_path, encoding='utf-8')

    X = np.array(df_pd['processed'])
    y = np.array(df_pd['sentimen'])

    cv = CountVectorizer()
    X = cv.fit_transform(X)

    # Initialize KFold with shuffle=True
    kf = KFold(n_splits=2, random_state=42, shuffle=True)
    print(kf)  # Display KFold parameters
    i = 1
    for train_index, test_index in kf.split(X):
        print("Fold ", i)
        print("TRAIN :", train_index, "TEST :", test_index)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        i += 1
    print("shape x_train :", X_train.shape)
    print("shape x_test :", X_test.shape)

    from sklearn import model_selection, svm
    from sklearn.svm import LinearSVC

    clf = svm.SVC(C=1.0, kernel='linear', degree=1, gamma='auto')
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

    content = data
    y_pred = clf.predict(X_test)
    actual = y_test
    predicted = y_pred
    results = confusion_matrix(actual, predicted)
    print('Confusion Matrix :')
    print(results)
    akurasi = accuracy_score(y_pred, y_test)*100

    return render_template('prediction_new.html',content=message,prediction=my_prediction,akurasi=akurasi)

@app.route('/get_map')
def get_map():
    r = int(random.triangular(0,100))
    t = "templates/map_{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("html/map_cluster.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

@app.route('/get_point')
def get_point():
    r = int(random.triangular(0,100))
    t = "templates/point{i}.html"
    for i in range(0,100):
        f = t.format(i=i)
        if os.path.exists(f):
            os.remove(f)
    f = t.format(i=r)
    shutil.copy("html/point.html", f)

    r = make_response(render_template(os.path.split(f)[1]))
    r.cache_control.max_age = 0
    r.cache_control.no_cache = True
    r.cache_control.no_store = True
    r.cache_control.must_revalidate = True
    r.cache_control.proxy_revalidate = True
    return r

if __name__ == '__main__':
    app.run(debug=True)
