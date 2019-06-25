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

  def normalizarDados(self, file):
    # Le arquivo de entrada e normaliza dados
    inputFile = open(file, 'r')
    data = inputFile.readlines()
    inputFile.close()

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

  def treinamento(self, file_treino, file_teste, batchSize, epochs, erroMax,
                  epochsMax):
    self.batchSize = batchSize
    self.epochs = 0

    data, labels = self.normalizarDados(file_treino)

    # Particiona os dados em treinamento e validacao
    (train_x, test_x, train_y, test_y) = train_test_split(
        data, labels, test_size=0.25)

    train_y = to_categorical(train_y, num_classes=self.nClasses)
    test_y = to_categorical(test_y, num_classes=self.nClasses)

    # Mostra o modelo criado
    self.model.summary()

    # ----------------------------------
    # Processo de treinamento do modelo
    # ----------------------------------
    adata, alabels = self.normalizarDados(file_teste)

    (atest_x, _, atest_y, _) = train_test_split(adata, alabels, test_size=1)

    atest_y = to_categorical(atest_y, num_classes=self.nClasses)
    erro = (100 - self.teste(atest_x, atest_y)[1] * 100
           )  # em porcentagem [0~100],

    while erro > erroMax or self.epochs < epochsMax:
      self.train_result = self.model.fit(
          train_x,
          train_y,
          validation_data=(test_x, test_y),
          epochs=epochs,
          batch_size=batchSize,
          verbose=2)
      erro = (100 - self.teste(atest_x, atest_y)[1] * 100)
      self.epochs += 1
      print("Erro de: %.2f%%" % erro)

    self.resultados(atest_x, atest_y)

  def teste(self, test_x, test_y):
    return self.model.evaluate(test_x, test_y, verbose=1)

  def resultados(self, test_x, test_y):
    test_y = np.argmax(test_y, axis=1)

    # Plot modelo
    plot_model(
        self.model,
        to_file='plot/model.png',
        show_shapes=True,
        show_layer_names=True)

    # Prever classes
    y_pred = self.model.predict_classes(test_x, batch_size=self.batchSize)

    # Reporte de Classificacao
    print("\n Reporte: \n", classification_report(test_y, y_pred, digits=10))

    # Matriz de Confusao
    cnf_matrix = confusion_matrix(test_y, y_pred)
    print("\n Matriz de Confusão: \n", cnf_matrix)

    # Plota a perda e as metricas
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, self.epochs),
        self.train_result.history["loss"],
        label="train_loss")
    plt.plot(
        np.arange(0, self.epochs),
        self.train_result.history["val_loss"],
        label="val_loss")
    plt.plot(
        np.arange(0, self.epochs),
        self.train_result.history["acc"],
        label="train_acc")
    plt.plot(
        np.arange(0, self.epochs),
        self.train_result.history["val_acc"],
        label="val_acc")
    plt.xlabel("Epoch " + str(self.epochs))
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.title("Training Loss and Accuracy Adam")
    plt.savefig("plot/fashion_mnist_Adam_01.png")
