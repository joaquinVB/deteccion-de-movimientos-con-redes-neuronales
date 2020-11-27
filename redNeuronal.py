import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from keras.models import model_from_yaml
from collections import Counter
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#volcamos los datos de excel en python
excel = pd.ExcelFile('path excel')
conjuntoDatos = excel.parse('conjuntoFinal')
seed = 7
np.random.seed(seed)


#dividimos los datos en set entrenamiento y set de validacion 
data = conjuntoDatos.iloc[1:231,1:13]
X = data.iloc[0:231,0:8]
Y = data.iloc[0:231,8:9]
trainingDataSetX, testDataSetX, trainingDataSetY, testDataSetY = train_test_split(X, Y, test_size=0.2, random_state=seed)




# codificamos target

#Entrenamiento
encoderTrainning = LabelEncoder()
encoderTrainning.fit(trainingDataSetY)
encoded_YTrainning = encoderTrainning.transform(trainingDataSetY)
# codificacion en caliente ( hot encoded)
dummy_yTrainning = to_categorical(encoded_YTrainning)



#Test
encoder = LabelEncoder()
encoder.fit(testDataSetY)
encoded_Y = encoder.transform(testDataSetY)
# codificacion en caliente ( hot encoded)
dummy_y = to_categorical(encoded_Y)



# crea el modelo
model = Sequential()
model.add(Dense(12, input_dim=8, init='normal', activation='relu'))
model.add(Dense(12,init='normal', activation='relu'))
model.add(Dense(8,init='normal', activation='sigmoid'))
# Compila el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["acc"])
# Ajusta el modelo
model_train = model.fit(trainingDataSetX, dummy_yTrainning, epochs=500, batch_size=2, verbose=1, validation_data=(testDataSetX, dummy_y))

validation_acc = model_train.history['val_acc'][-1] * 100
training_acc = model_train.history['acc'][-1] * 100
print("Validation accuracy: {}%\nTraining Accuracy: {}%".format(validation_acc, training_acc))




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')

    plt.show()


def plot_loss_and_accuracy(model_train):
  accuracy = model_train.history['acc']
  val_accuracy = model_train.history['val_acc']
  loss = model_train.history['loss']
  val_loss = model_train.history['val_loss']
  epochs = range(len(accuracy))
  plt.plot(epochs, accuracy, 'b', label='Training accuracy')
  plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
  plt.ylim(ymin=0)
  plt.ylim(ymax=1)
  plt.xlabel('Epochs ', fontsize=16)
  plt.ylabel('Accuracity', fontsize=16)
  plt.title('Training and validation accuracy', fontsize = 20)
  plt.legend()
  plt.figure()
  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.xlabel('Epochs ',fontsize=16)
  plt.ylabel('Loss',fontsize=16)
  plt.title('Training and validation loss', fontsize= 20)
  plt.legend()
  plt.show()


plot_loss_and_accuracy(model_train)


# serializa el modelo para YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
  yaml_file.write(model_yaml)
# serializa los pesos(weights) para HDF5
model.save_weights("model.h5")

print("Modelo guardado al PC")
# despues...
#carga del YAML y crea el modelo
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# cargamos los pesos (weights) en el nuevo modelo
loaded_model.load_weights("model.h5")
print("Modelo cargado desde el PC")
# evalua el modelo con los datos test
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])



predictions = model.predict(testDataSetX)


# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(predictions, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(dummy_y, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(8))

#obtenemos metricas de rendimiento
metricas = classification_report(Y_true, Y_pred_classes) 
print(metricas)
