import csv
import numpy as np
from scipy import stats


# step1: read csv file indians-diabetes.txt total 768 data
def loadDataSet():
    allData = open('/Users/lihaotian/PycharmProjects/homework/indians-diabetes.txt', 'r')
    reader = csv.reader(allData)
    # print headerLine
    header = reader.next()
    # print("\n header: \n" + str(header))
    # print dataSet save as a array/mat format
    dataSetWithoutHeader = []
    for row in reader:
        dataSetWithoutHeader.append(row)
    # print np.array(dataSetWithoutHeader)
    # print np.mat(dataSetWithoutHeader)
    return header, dataSetWithoutHeader



# step2: classify the outcomeLabels and split trainingSet and testSet
def splitDataSet(dataSet):
    m, n = np.shape(dataSet)
    classLabels = np.array([example[-1] for example in dataSet])
    # print '\n classLabels: \n', classLabels

    trainingSet = range(len(dataSet))
    testSet = []
    for i in range(int(len(trainingSet)*1/3)):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    # print("\n trainingSet: \n" + str(trainingSet))
    # print("\n testSet: \n" + str(testSet))

    trainMat = []
    trainingClassMat = []
    for index in trainingSet:
        trainMat.append(dataSet[index])
        # print("\n trainMat: \n" + str(np.array(trainMat)))
        # print("\n shape: \n" + str(np.shape(trainMat)))
        trainingClassMat.append(classLabels[index])
    # print("\n trainMat: \n" + str(np.array(trainMat)))
    # print("\n shape: \n" + str(np.shape(trainMat)))
    # print("\n trainingClassMat: \n" + str(np.array(trainingClassMat)))
    # print("\n shape: \n" + str(np.shape(trainingClassMat)))

    testMat = []
    testClassMat = []
    for index in testSet:
        testMat.append(dataSet[index])
        # print("\n testMat: \n" + str(np.array(testMat)))
        # print("\n shape: \n" + str(np.shape(testMat)))
        testClassMat.append(classLabels[index])
    # print("\n testMat: \n" + str(np.array(testMat)))
    # print("\n shape: \n" + str(np.shape(testMat)))
    # print("\n testClassMat: \n" + str(np.array(testClassMat)))
    # print("\n shape: \n" + str(np.shape(testClassMat)))

    testMatNoClassLabels =[]
    for index in testMat:
        del (index[n-1])
        testMatNoClassLabels.append(index)

    # print("\n testMatNoClassLabel: \n" + str(np.array(testMatNoClassLabels)))

    return trainMat, trainingClassMat, testMatNoClassLabels, testClassMat, testSet
    # return trainMat, trainingClassMat, trainingSet, testSet



# step3: spilt the trainMat to 2 class(0 & 1)
def splitTrsinMat2class(trainMat):
    m, n = np.shape(trainMat)
    trainClassMat0 = []
    trainClassMat1 = []
    for index in trainMat:
        if index[n-1] == '0':
            trainClassMat0.append(index)
        else:
            trainClassMat1.append(index)
    # print("\n trainClassMat0: \n" + str(np.array(trainClassMat0)))
    # print("\n shape: \n" + str(np.shape(trainClassMat0)))
    # print("\n trainClassMat1: \n" + str(np.array(trainClassMat1)))
    # print("\n shape: \n" + str(np.shape(trainClassMat1)))
    return trainClassMat0, trainClassMat1



# step4: calculate the meanValue and variance of every feature(total calculate 8)
def calculateMeanAndVar(dataSet):
    # get every feature of input dataSet
    m,n = np.shape(dataSet)
    featureValues = []
    for i in range(n-1):
        featureValues.append([example[i] for example in dataSet])
    # print("\n featureValues: \n" + str(np.array(featureValues)))
    # print("\n shape: \n" + str(np.shape(np.array(featureValues))))

    #calculate meanValue and variance
    meanValueResult = []
    varianceResult = []
    for featureNum in range(n-1):
        featureArray = np.array(featureValues[featureNum]).astype(float)
        meanValueResult.append(np.mean(featureArray))
        varianceResult.append(np.var(featureArray))
    # print("\n meanValueResult: \n" + str(meanValueResult))
    # print("\n varianceResult: \n" + str(varianceResult))
    # print("\n featureValues[1]: \n" +str(featureValues[1]))
    # print("\n shape: \n" + str(np.shape(featureValues[1])))

    return meanValueResult, varianceResult, n-1



# step5: return 8 Probability Density Function(PDFunction)
def PDFunction(x, meanValue, variance, featureCounts):
    PDFunctionResults = []
    for i in range(featureCounts):
        PDFunctionResults.append(stats.norm.pdf(x[i], meanValue[i], variance[i]))

    return PDFunctionResults



#step6: trainingNB
def classifyNB(trainMat, trainingClassMat, testMat):
    pClass1 = sum([int(i) for i in trainingClassMat])/float(len(trainMat))
    # print("\npClass1: " + str(pClass1))

    # step3
    trainClassMat0, trainClassMat1 = splitTrsinMat2class(trainMat)

    # step4
    meanValueResult0, varianceResult0, featureCounts0 = calculateMeanAndVar(trainClassMat0)
    meanValueResult1, varianceResult1, featureCounts1 = calculateMeanAndVar(trainClassMat1)

    # step5
    testResult = []
    for indexSet in testMat:
        testX = [float(i) for i in indexSet]
        # print testX
        PDFunctionResults0 = PDFunction(testX, meanValueResult0, varianceResult0, featureCounts0)
        PDFunctionResults1 = PDFunction(testX, meanValueResult1, varianceResult1, featureCounts1)

        # Data processing: add log, Prevent data too small and the result overflow
        pPDFunctionResults0 = np.log(PDFunctionResults0)
        pPDFunctionResults1 = np.log(PDFunctionResults1)
        # print("\nPDFunctionResults0: \n" + str(pPDFunctionResults0))
        # print("\nPDFunctionResults1: \n" + str(pPDFunctionResults1))

        p0 = sum(pPDFunctionResults0) + np.log(1.0 - pClass1)
        p1 = sum(pPDFunctionResults1) + np.log(pClass1)
        # print("\n sum(pPDFunctionResults0): \n" + str(sum(pPDFunctionResults0)))
        # print("\n sum(pPDFunctionResults1): \n" + str(sum(pPDFunctionResults1)))

        if p1 > p0:
            testResult.append(1)
        else:
            testResult.append(0)
    # print("\n testResultl: \n" + str(testResult))

    return testResult



#step7: accuracy rate
def accuracyRate(testClassMat):
    testResult = classifyNB(trainMat, trainingClassMat, testMatNoClassLabels)

    actualResult1 = []
    for index in testClassMat:
        actualResult1.append(index)
    actualResult = [int(i) for i in actualResult1]

    # print("\n testResult: \n" + str(testResult))
    # print("\n actualResult: \n" + str(actualResult))

    errorCount = 0
    for i in range(len(testClassMat)):
        if actualResult[i] != testResult[i]:
            errorCount += 1
    # print errorCount
    errorRate = float(errorCount)/len(testClassMat)

    # print("\n errorRate: " + str(errorRate*100) + "%")
    print("\n accuracyRate: " + str(100 - errorRate * 100) + "%")
    return errorRate



if __name__ == '__main__':
    # step1
    header, dataSetNoHeader = loadDataSet()

    # step2
    # trainMat, trainingClassMat, trainingSet, testSet = splitDataSet(dataSetNoHeader)
    trainMat, trainingClassMat, testMatNoClassLabels, testClassMat, testSet = splitDataSet(dataSetNoHeader)

    # step6
    # classifyNB(trainMat, trainingClassMat, testMatNoClassLabels)
    accuracyRate(testClassMat)
