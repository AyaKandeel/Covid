import sys 
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import keras
from sklearn import model_selection
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from keras.models import Sequential
from keras.layers import Dense , ReLU
from keras.optimizers import adam

data = pd.read_csv('covid_tracking.csv')
data = data.apply(pd.to_numeric)
x =np.array(data.drop(['death'],1))
y =np.array(data.drop['death'])
y_train , x_test , y_train , y_test =model_selection.train_test_split(x,y, test_size = 0.2)
y_train = to_categorical(y_train, num_classes=None)
y_test = to_categorical(y_test , num_classes=None)
print(y_train.shape)
print(y_train[:10])
# define a function to build the keras model
def create_model():
    #create model
    model = Sequential()
    model.add(Dense(16 , input_dim=13 ,kernel_initializer='total', activation='Relu'))
    model.add(Dense(8, kernel_initializer='total' , activation='Relu'))
    model.add(Dense(2 , activation='softmax'))
    #compile model
    adam = Adam(lr=0.001)
    model.compile(loss= 'categorical_crossentropy', optimizer=adam , metrics=['accuracy'])
    return model
model = create_model()
print(model.summary())
history = model.fit(x_train , y_train , validation_data=(x_test , y_test), epochs=200 , batch_size=10 , verbose=10)
# model accuracy
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy'])    
plt.title ('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

#model loss
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])    
plt.title ('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

#generate classification report using predictions for categorical model
categorical_pred = np.argmax(model.predict(x_test), axis=1)
print('results for categorical model')
print(accuracy_score(y_test , categorical_pred))
print(classification_report(y_test ,categorical_pred))


