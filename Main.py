from Training import Training
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
  width = 28
  height = 28
  nClasses = 10
  lr = 1e-2
  batchSize = 100
  epochs = 20

  # criando modelo
  model = NeuralNetwork().criar(nClasses, width*height, lr)
  
  # treinando modelo
  t = Training(model, nClasses)
  t.train("./dados/fashion-mnist_train.csv", batchSize, epochs)
