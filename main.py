from data_utils import *
import numpy as np

# * load Data from the files
mathDataHeaders, mathDataList = loadData('../student-data/student-mat.csv')
portugalDataHeaders, portugalDataList = loadData('../student-data/student-por.csv')

# print(mathDataHeaders)
# print(portugalDataHeaders)

# * shuffle the data lists -- this could be optional, i think
shuffleData(mathDataList)
shuffleData(portugalDataList)

# * divide the data into sub samples: 40% training, 30% validation, 30% testing
mathTraining, mathValidation, mathTesting = subSamples(mathDataList, 0.40, 0.30)
portugalTraining, portugalValidation, portugalTesting = subSamples(portugalDataList, 0.40, 0.30)

# * from the data lists, we separate the output column (G3) 
mathTraining, mathTrainingTargets = getTargetOutputs(mathTraining)
mathValidation, mathValidationTargets = getTargetOutputs(mathTraining)
mathTesting, mathTestingTargets = getTargetOutputs(mathTraining)

portugalTraining, portugalTrainingTargets = getTargetOutputs(mathTraining)
portugalValidation, portugalValidationTargets = getTargetOutputs(mathTraining)
portugalTesting, portugalTestingTargets = getTargetOutputs(mathTraining)

# print(len(mathTraining),len(mathValidation), len(mathTesting))
# print(len(portugalTraining),len(portugalValidation), len(portugalTesting))

# test = initializeWeights()
# print(test)
# * initialize the weights for the hidden nodes (33-23-1 network: 33x23 weights + 23 bias weights = 782)
# hiddenNodeWeights = [ initializeWeights() for _ in range(759)]
hiddenNodeWeights = np.random.uniform(0.0, 0.1, (23, 33)) # 33 weights for the 23 hidden nodes
hiddenNodeBiasWeights = np.random.uniform(0.0, 0.1, 23) # bias weights for each hidden node

# print(hiddenNodeWeights)
# print(hiddenNodeBiasWeights)

# * initialize the weights for the output node (1 output node: 23 hidden nodes connecting + 1 bias weight)
# half of the nodes will be 1 and the other half will be -1, since there is odd number, bias node is 0
outputNodeWeights = [
  1,1,1,1,1,1,
  1,1,1,1,1,1,
  -1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1
]
outputNodeBiasWeight = 0


