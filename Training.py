import matplotlib
matplotlib.use("Agg")
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model


class Training:

  def __init__(self, model, nClasses):
    self.nClasses = nClasses
    self.model = model

  def normalizarDados(self, data):
    dataset = []
    labels = []

    for i, item in enumerate(data):
      if i == 0:
        continue

      values = item.split(',')

      # Normaliza de 0.01 ate 1
      dataset.append((np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01)
      labels.append(int(values[0]))

    return np.array(dataset), np.array(labels)

  def train(self, file, batchSize, epochs):
    # Le arquivo de entrada e normaliza dados
    inputFile = open(file, 'r')
    data = inputFile.readlines()
    inputFile.close()

    data, labels = self.normalizarDados(data)

    # Particiona os dados em treinamento(75%) e validação(25%)
    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25)
    train_y = to_categorical(train_y, num_classes=self.nClasses)
    test_y = to_categorical(test_y, num_classes=self.nClasses)

    # Mostra o modelo criado
    self.model.summary()

    # ----------------------------------
    # Processo de treinamento do modelo
    # ----------------------------------
    H = self.model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        epochs=epochs,
        batch_size=batchSize,
        verbose=2)

    pont = self.model.evaluate(test_x, test_y, verbose=1)
    print("Erro de: %.2f%%", (100 - pont[1] * 100))

    #report(self.model, test_x, test_y)