import random

def loadData(fileDirectory):
  # dataFile = open('../student-data/student-mat.csv')
  # get file name and open it
  dataFile = open(fileDirectory)
  #extract headers and turn into list
  dataHeaders = dataFile.readline()
  dataHeaders = dataHeaders.split(';')
  # remove \n in the last element
  lastElement = dataHeaders[-1]
  dataHeaders[-1] = lastElement[0:(len(lastElement)-1)]

  dataList = []
  for data in dataFile: # iterate through each line
    dataInLineHolder = data.split(';') # convert a line of data into a list
    # remove \n in the last element
    lastElement = dataInLineHolder[-1]
    dataInLineHolder[-1] = lastElement[0:(len(lastElement)-1)]
    dataList.append(dataInLineHolder) # add to data list


  return dataHeaders, dataList

def shuffleData(dataList):
  random.shuffle(dataList)

def subSamples(dataList, trainFactor, validFactor):
  dataListLen = len(dataList)
  dataListTrainingEndIndex = int(trainFactor*dataListLen)
  dataListValidEndIndex = int(validFactor*dataListLen)
  # the remaining data in the list will be in test data list

  return dataList[0:dataListTrainingEndIndex], dataList[dataListTrainingEndIndex:(dataListTrainingEndIndex+dataListValidEndIndex)], dataList[(dataListTrainingEndIndex+dataListValidEndIndex):dataListLen]
