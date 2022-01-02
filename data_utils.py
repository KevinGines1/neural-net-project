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

  encodedDataList = encodeData(dataList, dataHeaders) #* represent the data into values to be plugged in to the network

  return dataHeaders, encodedDataList

def shuffleData(dataList):
  random.shuffle(dataList)

def subSamples(dataList, trainFactor, validFactor):
  dataListLen = len(dataList)
  dataListTrainingEndIndex = int(trainFactor*dataListLen)
  dataListValidEndIndex = int(validFactor*dataListLen)
  # the remaining data in the list will be in test data list

  return dataList[0:dataListTrainingEndIndex], dataList[dataListTrainingEndIndex:(dataListTrainingEndIndex+dataListValidEndIndex)], dataList[(dataListTrainingEndIndex+dataListValidEndIndex):dataListLen]

def encodeData(dataList, dataHeaders):
  #*this is a bit of a hardcoded way of transforming the data into values for the network. 
    #* it only applies to this specific dataset
  encodedDataList = []
  encodedData = []
  for data in dataList: 
    for i in range(len(data)):
      if(dataHeaders[i] == 'school'):
        if(data[i] == 'GP'):
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # for MS
      elif(dataHeaders[i] == 'sex'):
        if(data[i] == 'M'):
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # for female
      elif(dataHeaders[i] == 'address'):
        if(data[i] == 'U'): # urban
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # for rural
      elif(dataHeaders[i] == 'famsize'):
        if(data[i] == 'LE3'): # less or equal to 3
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # for greater than 3
      elif(dataHeaders[i] == 'Pstatus'):
        if(data[i] == 'T'): # together
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # for apart
      elif(dataHeaders[i] == 'Mjob' or dataHeaders[i] == 'Fjob'):
        if(data[i] == 'teacher'):
          encodedData.append(0.1)
        elif(data[i] == 'health'):
          encodedData.append(1)
        elif(data[i] == 'services'):
          encodedData.append(2)
        elif(data[i] == 'at_home'):
          encodedData.append(3)
        else:
          encodedData.append(4) # other
      elif(dataHeaders[i] == 'reason'):
        if(data[i] == 'home'): # close to home
          encodedData.append(0.1)
        elif(data[i] == 'reputation'): # school reputation
          encodedData.append(1)
        elif(data[i] == 'course'): # course preference
          encodedData.append(2)
        else:
          encodedData.append(3) # other
      elif(dataHeaders[i] == 'guardian'):
        if(data[i] == 'mother'):
          encodedData.append(0.1)
        elif(data[i] == 'father'):
          encodedData.append(1)
        else:
          encodedData.append(2) # other
      elif(dataHeaders[i] in ['schoolsup','famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']):
        if(data[i] == 'no'):
          encodedData.append(0.1)
        else:
          encodedData.append(1) # yes
      else:
        try:
          encodedData.append(int(data[i])) # value of data need not to be encoded
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          encodedData.append(int(cleanerString))
    encodedDataList.append(encodedData)
    encodedData = []
  return encodedDataList

def getTargetOutputs(dataList):
  targetColumn = []
  for data in dataList: 
    targetColumn.append(data.pop())
  return dataList, targetColumn