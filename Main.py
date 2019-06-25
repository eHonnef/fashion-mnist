from Training import Training
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
  width = 28        # Largura das imagens
  height = 28       # Altura das imagens
  nClasses = 10     # Numero de classes (numero de itens)
  lr = 1e-2         #
  batchSize = 100   #
  epochs = 20       # numero de epocas
  epochsMax = 20    # numero maximo de epocas
  erroMax = 10      # erro maximo (em porcentagem)

  # criando modelo
  model = NeuralNetwork().criar(nClasses, width*height, lr)
  
  # treinando modelo
  t = Training(model, nClasses)
  t.treinamento("./dados/fashion-mnist_train.csv", "./dados/fashion-mnist_test.csv", batchSize, epochs, erroMax, epochsMax)
