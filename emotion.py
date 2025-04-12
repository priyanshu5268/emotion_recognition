# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

from sklearn import datasets, tree, linear_model, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Flatten
from tensorflow.keras.utils import to_categorical

# Clear previous TensorFlow sessions
tf.keras.backend.clear_session()

# Reading the EEG emotion dataset
data = pd.read_csv("D:/data/emotion.csv")

# Displaying the number of samples
print("Total Samples:", len(data))

# Splitting the data into positive, negative, and neutral for visualization
pos = data.loc[data["label"] == "POSITIVE"]
neg = data.loc[data["label"] == "NEGATIVE"]
neu = data.loc[data["label"] == "NEUTRAL"]

sample_pos = pos.loc[2, 'fft_0_b':'fft_749_b']
sample_neg = neg.loc[0, 'fft_0_b':'fft_749_b']
sample_neu = neu.loc[1, 'fft_0_b':'fft_749_b']

# Plotting label distribution
plt.figure(figsize=(25, 7))
plt.title("Data Distribution of Emotions")
sns.countplot(x='label', data=data)
plt.show()

# Plotting sample signal for POSITIVE emotion
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample_pos)), sample_pos)
plt.title("Positive Emotion EEG Signal")
plt.show()

# Plotting sample signal for NEGATIVE emotion
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample_neg)), sample_neg)
plt.title("Negative Emotion EEG Signal")
plt.show()

# Plotting sample signal for NEUTRAL emotion
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample_neu)), sample_neu)
plt.title("Neutral Emotion EEG Signal")
plt.show()

# Data transformation and preprocessing function
def Transform_data(data):
    # Label encoding
    encoding_data = {'NEUTRAL': 0, 'POSITIVE': 1, 'NEGATIVE': 2}
    data_encoded = data.replace(encoding_data)

    # Features and labels
    x = data_encoded.drop(["label"], axis=1)
    y = data_encoded['label'].values

    # Feature scaling
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # One-hot encoding the labels
    y_encoded = to_categorical(y)

    return x_scaled, y_encoded

# Transforming and splitting the dataset
X, Y = Transform_data(data)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

# Model creation using GRU
def create_model():
    inputs = tf.keras.Input(shape=(x_train.shape[1],))
    expand_dims = tf.expand_dims(inputs, axis=2)
    gru = tf.keras.layers.GRU(256, return_sequences=True)(expand_dims)
    flatten = tf.keras.layers.Flatten()(gru)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(flatten)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

# Compile and train the model
lstmmodel = create_model()
lstmmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = lstmmodel.fit(x_train, y_train, epochs=10, validation_split=0.1)

# Evaluate the model
loss, acc = lstmmodel.evaluate(x_test, y_test)
print(f"Loss on testing: {loss * 100:.2f}%")
print(f"Accuracy on testing: {acc * 100:.2f}%")

# Model predictions
pred = lstmmodel.predict(x_test)
pred1 = np.argmax(pred, axis=1)
y_test1 = np.argmax(y_test, axis=1)

# Print some predictions vs actual
print("Predicted:  ", pred1[:10])
print("Actual:     ", y_test1[:10])

# Confusion matrix function
def plot_confusion_matrix(cm, names, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=90)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# Plotting LSTM confusion matrix
cm = confusion_matrix(y_test1, pred1)
plt.rcParams["figure.figsize"] = (10, 5)
plt.figure()
plot_confusion_matrix(cm, ["Neutral", "Positive", "Negative"])
plt.show()

# Evaluate other classifiers
print('\n*\t\tClassification Report - LSTM:\n', classification_report(y_test1, pred1))

# Train Gaussian Naive Bayes
Classifier_gnb = GaussianNB().fit(x_train, np.argmax(y_train, axis=1))
pred_gnb = Classifier_gnb.predict(x_test)
print('\n*\t\tClassification Report - GaussianNB:\n', classification_report(y_test1, pred_gnb))

# Train SVM
Classifier_svm = svm.SVC(kernel='linear').fit(x_train, np.argmax(y_train, axis=1))
pred_svm = Classifier_svm.predict(x_test)
print('\n*\t\tClassification Report - SVM:\n', classification_report(y_test1, pred_svm))

# Train Logistic Regression
Classifier_LR = linear_model.LogisticRegression(solver='liblinear', C=75).fit(x_train, np.argmax(y_train, axis=1))
pred_LR = Classifier_LR.predict(x_test)
print('\n*\t\tClassification Report - Logistic Regression:\n', classification_report(y_test1, pred_LR))

# Train Decision Tree
Classifier_dt = tree.DecisionTreeClassifier().fit(x_train, np.argmax(y_train, axis=1))
pred_dt = Classifier_dt.predict(x_test)
print('\n*\t\tClassification Report - Decision Tree:\n', classification_report(y_test1, pred_dt))

# Train Random Forest
Classifier_forest = RandomForestClassifier(n_estimators=50, random_state=0).fit(x_train, np.argmax(y_train, axis=1))
pred_fr = Classifier_forest.predict(x_test)
print('\n*\t\tClassification Report - Random Forest:\n', classification_report(y_test1, pred_fr))

# Plotting confusion matrices of all classifiers
classifiers = [Classifier_gnb, Classifier_svm, Classifier_LR, Classifier_forest]
names1 = ["Neutral", "Positive", "Negative"]
colors = ['YlOrBr', 'GnBu', 'Pastel2', 'PuRd']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
for cls, ax, c in zip(classifiers, axes.flatten(), colors):
    plot_confusion_matrix(cls, x_test, y_test1, ax=ax, cmap=c, display_labels=names1)
    ax.title.set_text(type(cls).__name__)
plt.tight_layout()
plt.show()

# Plot training loss and accuracy of LSTM model
plt.style.use("fivethirtyeight")
plt.figure(figsize=(20, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss", fontsize=20)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy", fontsize=20)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.show()
