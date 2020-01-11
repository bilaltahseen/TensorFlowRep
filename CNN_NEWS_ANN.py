import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras import models, layers


def iniitalized():
    data = pd.read_csv("bbc-text.csv")
    # data.head()
    data['category'].value_counts()
    X = data['text']
    y = data['category']
    X_train, X_test, y_train_cat, y_test_cat = train_test_split(
        X, y, test_size=0.20, random_state=10)
    # Train test split
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train_cat.reset_index(drop=True, inplace=True)
    y_test_cat.reset_index(drop=True, inplace=True)
    max_words = 1000
    tokenize = keras.preprocessing.text.Tokenizer(
        num_words=max_words, char_level=False)
    tokenize.fit_on_texts(X_train)  # fit tokenizer to our training text data
    x_train = tokenize.texts_to_matrix(X_train)
    x_test = tokenize.texts_to_matrix(X_test)
    # x_train[0].shape
    encoder = LabelEncoder()
    encoder.fit(y_train_cat)
    y_train = encoder.transform(y_train_cat)
    y_test = encoder.transform(y_test_cat)
    # Converts the labels to a one-hot representation
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
   # print('x_train shape:', x_train.shape)
   # print('x_test shape:', x_test.shape)
   # print('y_train shape:', y_train.shape)
   # print('y_test shape:', y_test.shape)
    model = models.Sequential()
    model.add(layers.Dense(1024, input_shape=(max_words,)))
   # Use for relu in our model is that there isn't any negative dependacy in our dataset
    model.add(layers.Dense(212, activation='relu'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    batch_size = 32
    epochs = 2
    drop_ratio = 0.5
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, verbose=1, validation_split=0.1)
    score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
   #  print('Test loss:', score[0])
   #  print('Test accuracy:', score[1])
#    text_labels = encoder.classes_
#    for i in range(10):
#        print('Test loss:', score[0])
#        print('Test accuracy:', score[1])
#     print("----------------")
#     prediction = model.predict(np.array([x_test[i]]))
#     predicted_label = text_labels[np.argmax(prediction)]
#     print(X_test[i][:50], "...")
#     print('Actual label:' + y_test_cat[i])
#     print("Predicted label: " + predicted_label + "\n")
    text_labels = encoder.classes_
    text_inp = input("Enter News")
    max_words = 1000
    tokenize = keras.preprocessing.text.Tokenizer(
        num_words=max_words, char_level=False)
    X_train_inp = pd.Series(text_inp)
    # fit tokenizer to our training text data
    tokenize.fit_on_texts(X_train)
    x_train_man = tokenize.texts_to_matrix(X_train_inp)
    prediction = model.predict(np.array([x_train_man[0]]))
    predicted_label = text_labels[np.argmax(prediction)]
    print('----------------')
    print(f'The Predicted News Class is: {predicted_label}')


iniitalized()
