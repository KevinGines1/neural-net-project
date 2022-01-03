import numpy as np
# from decimal import Decimal, getcontext
# getcontext().prec=100


def sigmoidFunction(value):
  # this is dE/dz
  return 1/(1+np.exp(-value)) # logistic sigmoid function

def sigmoidFunction_deriv(value):
  # this is dz/dv 
  # this is dy/du
  return sigmoidFunction(value) - (1-sigmoidFunction(value))

def sumOfProducts(oneNodeWeights, values):
  # summation of aixi or bjyj
  # assumed that both lists are equal in length
  # sum = Decimal(0)
  sum = 0
  for i in range(len(oneNodeWeights)):
    # sum = Decimal(sum) + Decimal(oneNodeWeights[i])*Decimal(values[i])
    sum = sum + (oneNodeWeights[i]*values[i])
  # return Decimal(sum)
  return sum

def trainNeuralNetwork(numEpoch, learningRate, dataList, hiddenNodeWeights, hiddenNodeBiasWeights, outputNodeWeights, outputNodeBiasWeight, targetOutput):

  numHiddenNodes = len(hiddenNodeWeights)
  numInputNodes = len(dataList[0])
  u_values = np.zeros(numHiddenNodes) # array of u's
  y_values = np.zeros(numHiddenNodes) # array of y's


  for epoch in range(numEpoch): # iterate based on the number of epochs declared
    for sample in range(len(dataList)): # iterate thru the data list 
      for node in range(numHiddenNodes): # iterate thru the hiddenNodes to calculate u and y
        # u_values[node] = Decimal(hiddenNodeBiasWeights[node]) + sumOfProducts(hiddenNodeWeights[node], dataList[sample])
        u_values[node] = hiddenNodeBiasWeights[node] + sumOfProducts(hiddenNodeWeights[node], dataList[sample])
        # y_values[node] = Decimal(sigmoidFunction(u_values[node]))
        y_values[node] = sigmoidFunction(u_values[node])
        # print(len(y_values), len(outputNodeWeights))
      
      # calculate v and z
      v = outputNodeBiasWeight + sumOfProducts(outputNodeWeights, y_values)
      z = sigmoidFunction(v)

      #get dE/dz = z-t
      # FE = z - Decimal(targetOutput[sample])
      FE = z - targetOutput[sample]

      # backpropagation
      for hiddenNode in range(numHiddenNodes):
        # get dE/dz * dZ/dV (p)
        p = FE * sigmoidFunction_deriv(v) # this is also the outputNodeBiasWeightDeriv

        # the following two lines of code are dE/dbj
        # outputNodeWeightErrorDeriv = p * Decimal(y_values[hiddenNode])
        outputNodeWeightErrorDeriv = p * y_values[hiddenNode]

        for inputNode in range(numInputNodes):
          inputValue = dataList[sample][inputNode] # x1
          # get dE/dai = summation of pk*bk*y(1-y)*x1
          q = p * outputNodeWeights[hiddenNode] * sigmoidFunction_deriv(u_values[hiddenNode]) # this is also the hiddenNodeBiasWeightDeriv
          hiddenNodeWeightDeriv = q * inputValue

          #adjust the weights for the hidden node
          # hiddenNodeBiasWeights[hiddenNode] = Decimal(hiddenNodeBiasWeights[hiddenNode]) - (learningRate * q)
          hiddenNodeBiasWeights[hiddenNode] = hiddenNodeBiasWeights[hiddenNode] - (learningRate * q)
          # hiddenNodeWeights[hiddenNode][inputNode] = Decimal(hiddenNodeWeights[hiddenNode][inputNode]) - (learningRate * hiddenNodeWeightDeriv)
          hiddenNodeWeights[hiddenNode][inputNode] = hiddenNodeWeights[hiddenNode][inputNode] - (learningRate * hiddenNodeWeightDeriv)
        
        #adjust the weights for the output node
        # outputNodeBiasWeight = Decimal(outputNodeBiasWeight) - (learningRate * p)
        outputNodeBiasWeight = outputNodeBiasWeight - (learningRate * p)
        # outputNodeWeights[hiddenNode] = Decimal(outputNodeWeights[hiddenNode]) - (learningRate * outputNodeWeightErrorDeriv)
        outputNodeWeights[hiddenNode] = outputNodeWeights[hiddenNode] - (learningRate * outputNodeWeightErrorDeriv)

def validateNetWork(dataList, hiddenNodeWeights, hiddenNodeBiasWeights, outputNodeWeights, outputNodeBiasWeight, targetOutput):
  
  correctPredictionCount = 0
  numData = len(dataList)
  numHiddenNodes = len(hiddenNodeWeights)
  u_values = np.zeros(numHiddenNodes) # array of u's
  y_values = np.zeros(numHiddenNodes) # array of y's
  
  for sample in range(numData):
    for node in range(numHiddenNodes):
      # u_values[node] = Decimal(hiddenNodeBiasWeights[node]) + sumOfProducts(hiddenNodeWeights[node], dataList[sample])
      u_values[node] = hiddenNodeBiasWeights[node] + sumOfProducts(hiddenNodeWeights[node], dataList[sample])
      y_values[node] = sigmoidFunction(u_values[node])

    # v = Decimal(outputNodeBiasWeight) + sumOfProducts(outputNodeWeights, y_values)
    v = outputNodeBiasWeight + sumOfProducts(outputNodeWeights, y_values)
    # print(v)
    z = sigmoidFunction(v)

    if z > 0.5: 
      output = 1
    else: 
      output = 0
    
    if output == targetOutput[sample]:
      correctPredictionCount += 1
  
  return correctPredictionCount