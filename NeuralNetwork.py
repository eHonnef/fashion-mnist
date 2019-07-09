from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras import optimizers


class NeuralNetwork:

  # Cria e retorna uma rede neural
  @staticmethod
  def criar(nClasses, nPixels, lr):
    hf = int(nPixels / 2)  # primeira camada oculta
    hs = int(nPixels / 12)  # segunda camada oculta

    model = Sequential()
    model.add(
        Dense(
            nPixels,
            input_dim=nPixels,
            kernel_initializer='normal',
            activation='relu',
            name='input'))

    model.add(
        Dense(
            hf,
            input_dim=nPixels,
            kernel_initializer='normal',
            activation='relu',
            name='hidden_1'))

    model.add(
        Dense(
            hs,
            input_dim=hf,
            kernel_initializer='normal',
            activation='relu',
            name='hidden_2'))

    model.add(
        Dense(
            nClasses,
            kernel_initializer='normal',
            activation='softmax',
            name='preds'))

    # Algoritmos da documentacao do keras
    # Erro 11.54%
    # optimizer = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)

    # Erro 10.76%
    # optimizer = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)

    # Erro 17.28%
    # optimizer = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=0.0)

    # Erro 10.28%
    optimizer = optimizers.Adagrad(lr=lr, decay=1e-6)

    # Erro 17.63%
    # optimizer = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)

    # Erro 10.63%
    # optimizer = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

    # Erro de: 12.29%
    # optimizer = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    return model