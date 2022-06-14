# this file is for working on colab. if you use this code, please paste this on colab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import np_utils

from google.colab import drive
drive.mount('/gdrive')
data_train = pd.read_csv('/gdrive/My Drive/Colab Notebooks/mitbih_train.csv')
data_train

from google.colab import drive
drive.mount('/gdrive')
data_test = pd.read_csv('/gdrive/My Drive/Colab Notebooks/mitbih_train.csv')
data_test

x_train = data_train.iloc[:,0:187]
y_train= data_train.iloc[:,-1]
x_test= data_test.iloc[:,0:187]
y_test= data_test.iloc[:,-1]
y_train.value_counts()
y_test.value_counts()

# labeling 
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(y_train)

y_train_enc= encoder.transform(y_train)
y_test_enc= encoder.transform(y_test)

y_train_cat= np_utils.to_categorical(y_train_enc)
y_test_cat= np_utils.to_categorical(y_test_enc)

y_train_f= pd.DataFrame(y_train_cat)
y_train_f.value_counts()
y_test_f= pd.DataFrame(y_test_cat)
y_test_f.value_counts()
y_train_cat.value_counts()


# training
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()
model.add(Dense(128, input_dim=187, activation='relu'))
drop_rate = 0.03
for a in range(5):
  model.add(Dense(128, activation='relu'))
  model.add(keras.layers.Dropout(drop_rate))
    
model.add(Dense(5, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy'])


early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=50, # how many epochs to wait before stopping
    restore_best_weights=True,
)   

history = model.fit(
    x_train, y_train_f,
    validation_data=(x_test, y_test_f),
    batch_size=3000,
    epochs=250,
    callbacks=[early_stopping],
    verbose=1,  
)

history_df = pd.DataFrame(history.history)
# show history
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_accuracy'].max()))


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

preds_1 = model.predict(x_test)
preds_1_rounded = np.round(preds_1)

y_test_fn = np.array(y_test_f)

print(classification_report(np.argmax(y_test_fn, axis = 1), np.argmax(preds_1_rounded, axis = 1)))