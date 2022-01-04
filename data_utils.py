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
      elif(dataHeaders[i] == 'Mjob' or dataHeaders[i] == 'Fjob'): # normalize the values after representing the values
        if(data[i] == 'teacher'):
          encodedData.append(normalizeValue(0, 4, 0))
        elif(data[i] == 'health'):
          encodedData.append(normalizeValue(1, 4, 0))
        elif(data[i] == 'services'):
          encodedData.append(normalizeValue(2, 4, 0))
        elif(data[i] == 'at_home'):
          encodedData.append(normalizeValue(3, 4, 0))
        else:
          encodedData.append(normalizeValue(4, 4, 0))
      elif(dataHeaders[i] == 'reason'):
        if(data[i] == 'home'): # close to home
          encodedData.append(normalizeValue(0, 3, 0))
        elif(data[i] == 'reputation'): # school reputation
          encodedData.append(normalizeValue(1, 3, 0))
        elif(data[i] == 'course'): # course preference
          encodedData.append(normalizeValue(2, 3, 0))
        else:
          encodedData.append(normalizeValue(3, 3, 0))
      elif(dataHeaders[i] == 'guardian'):
        if(data[i] == 'mother'):
          encodedData.append(normalizeValue(0, 2, 0))
        elif(data[i] == 'father'):
          encodedData.append(normalizeValue(1, 2, 0))
        else:
          encodedData.append(normalizeValue(2, 2, 0))
      elif(dataHeaders[i] in ['schoolsup','famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']):
        if(data[i] == 'no'):
          encodedData.append(0.1)
        else:
          encodedData.append(0.99) # yes
      elif(dataHeaders[i] in ['freetime','goout', 'Dalc', 'Walc', 'health', 'famrel']): # values ranging 1-5 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 5, 1))
      elif(dataHeaders[i] in ['traveltime', 'studytime', 'failures']): # values ranging 1-4 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 4, 1))
      elif(dataHeaders[i] in ['Medu', 'Fedu']): # values ranging 0-4 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 4, 0))
      elif(dataHeaders[i] in ['G1', 'G2']): # values ranging 0-20 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 20, 0))
      elif(dataHeaders[i] == 'G3'): # convert to binary
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        if(value >= 10):
          encodedData.append(1) # PASS
        else: 
          encodedData.append(0) # FAIL
      elif(dataHeaders[i] == 'age'): # values ranging 15-22 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 22, 15))
      elif(dataHeaders[i] == 'absences'): # values ranging 0-93 will be normalized
        try:
          value = int(data[i])
        except:
          cleanString = data[i].lstrip('\'').rstrip('\'')
          cleanerString = data[i].lstrip('\"').rstrip('\"')
          value = int(cleanerString)
        encodedData.append(normalizeValue(value, 93, 0))
    
    encodedDataList.append(encodedData)
    encodedData = []
    
  return encodedDataList

def normalizeValue(value, Vmax, Vmin): # normalize the value to 0.1-0.9
  Tmax = 0.9 # max value of the target
  Tmin = 0.1 # min value of the target
  Tfactor = Tmax - Tmin
  V_denom = Vmax-Vmin

  return Tmin + (((value-Vmin)/V_denom)*Tfactor)
  
def getTargetOutputs(dataList):
  targetColumn = []
  for data in dataList: 
    targetColumn.append(data.pop()) # get the target output
    # Tar = normalizeValue(data.pop(), 20, 0) # get the raw target output
    # encode the target value
    # targetColumn.append(Tar)
  return dataList, targetColumn

