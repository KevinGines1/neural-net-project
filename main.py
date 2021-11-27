from loadData import *

# * load Data from the files
mathDataHeaders, mathDataList = loadData('../student-data/student-mat.csv')
portugalDataHeaders, portugalDataList = loadData('../student-data/student-por.csv')

# * shuffle the data lists -- this could be optional, i think
shuffleData(mathDataList)
shuffleData(portugalDataList)

# * divide the data into sub samples: 40% training, 30% validation, 30% testing
mathTraining, mathValidation, mathTesting = subSamples(mathDataList, 0.40, 0.30)
portugalTraining, portugalValidation, portugalTesting = subSamples(portugalDataList, 0.40, 0.30)

# print(len(mathTraining),len(mathValidation), len(mathTesting))
# print(len(portugalTraining),len(portugalValidation), len(portugalTesting))