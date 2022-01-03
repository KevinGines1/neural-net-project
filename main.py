from data_utils import *
from neuralNet_utils import *
import numpy as np

# * load Data from the files
mathDataHeaders, mathDataList = loadData('../student-data/student-mat.csv')
portugalDataHeaders, portugalDataList = loadData('../student-data/student-por.csv')

# print(mathDataHeaders)
# print(len(mathDataList[0]))
# print(portugalDataList)
# print(portugalDataHeaders)

# * shuffle the data lists -- this could be optional, i think
shuffleData(mathDataList)
shuffleData(portugalDataList)

# * divide the data into sub samples: 40% training, 30% validation, 30% testing
mathTraining, mathValidation, mathTesting = subSamples(mathDataList, 0.40, 0.30)
portugalTraining, portugalValidation, portugalTesting = subSamples(portugalDataList, 0.40, 0.30)

# * from the data lists, we separate the output column (G3) and encode them
# print(len(mathTraining[0]))
# print(mathTraining[0])

mathTraining, mathTrainingTargets = getTargetOutputs(mathTraining)
mathValidation, mathValidationTargets = getTargetOutputs(mathValidation)
mathTesting, mathTestingTargets = getTargetOutputs(mathTesting)

portugalTraining, portugalTrainingTargets = getTargetOutputs(portugalTraining)
portugalValidation, portugalValidationTargets = getTargetOutputs(portugalTraining)
portugalTesting, portugalTestingTargets = getTargetOutputs(portugalTraining)

# print(len(mathTraining[0]))
# print(mathTraining[0])
# print('==============\n',mathTrainingTargets)

# print(len(mathTraining),len(mathValidation), len(mathTesting))
# print(len(portugalTraining),len(portugalValidation), len(portugalTesting))

# test = initializeWeights()
# print(test)
# * initialize the weights for the hidden nodes (32-22-1 network: 32x22 weights + 22 bias weights = 726)
# hiddenNodeWeights = [ initializeWeights() for _ in range(759)]
hiddenNodeWeights = np.random.uniform(0.0, 0.1, (22, 32)) # 32 weights for the 23 hidden nodes
hiddenNodeBiasWeights = np.random.uniform(0.0, 0.1, 22) # bias weights for each hidden node

# print(hiddenNodeWeights)
# print(hiddenNodeBiasWeights)

# * initialize the weights for the output node (1 output node: 23 hidden nodes connecting + 1 bias weight)
# half of the nodes will be 1 and the other half will be -1, since there is odd number, bias node is 0
outputNodeWeights = [
  1,1,1,1,1,1,
  1,1,1,1,1,-1,
  -1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1
]
outputNodeBiasWeight = 0.01

# * initialize other needed values for the neural network
numEpoch = 1
learningRate = 0.4 # as good as gone haha

# ! TRAINING
# * MATHEMATICS
# print(hiddenNodeWeights[0])
# print(hiddenNodeBiasWeights[0])
# print(outputNodeWeights[0])
# print(outputNodeBiasWeight)
trainNeuralNetwork(numEpoch, learningRate, mathTraining, hiddenNodeWeights, hiddenNodeBiasWeights, outputNodeWeights, outputNodeBiasWeight, mathTrainingTargets)
# print(hiddenNodeWeights[0])
# print(hiddenNodeBiasWeights[0])
# print(outputNodeWeights[0])
# print(outputNodeBiasWeight)



# ! VALIDATING
correctPredictionCount = 0
correctPredictionCount = validateNetWork(mathValidation, hiddenNodeWeights, hiddenNodeBiasWeights, outputNodeWeights, outputNodeBiasWeight, mathValidationTargets)
print(correctPredictionCount)

# TODO: ENCODE THE OUTPUTS (TARGET ARRAYS FOR EACH DATASET) YOU'RE FUCKING ALMMOST THERE BITCH. HOLY FUCKING SHIT.

