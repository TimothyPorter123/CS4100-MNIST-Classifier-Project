import numpy

class Bayes_Classifier(object):

    def __init__(self):
        self.probabilityModel = numpy.zeros((10, 28, 28))


    def train(self, trainingData, laplacianSmoothing : int):

        bayesOccurances = numpy.zeros((10, 28, 28))
        bayesGroupOccurances = numpy.zeros(10)

        for i in range(0, len(trainingData[0])):
            print("Training on image: " + str(i))
            img = trainingData[0][i]
            imgVal = trainingData[1][i]
            bayesGroupOccurances[imgVal] = bayesGroupOccurances[imgVal] + 1
            for x in range(0, 28):
                for y in range(0, 28):
                    if img[x][y] > 127:
                        bayesOccurances[imgVal][x][y] = bayesOccurances[imgVal][x][y] + 1


        for i in range(0, 10):
            for x in range(0, 28):
                for y in range(0, 28):
                    bayesOccurances[i][x][y] = bayesOccurances[i][x][y] + laplacianSmoothing
                    self.probabilityModel[i][x][y] = bayesOccurances[i][x][y] / (bayesGroupOccurances[i] + laplacianSmoothing)
        return self

    def classify(self, img):
        classifications = numpy.zeros(10)
        for i in range(0, 10):
            classifierProb = 1
            for x in range(0, 28):
                for y in range(0, 28):
                    if(img[x][y] > 127):
                        unitProb = self.probabilityModel[i][x][y]
                    else:
                        unitProb = 1 - self.probabilityModel[i][x][y]
                    classifierProb = classifierProb * unitProb
                classifications[i] = classifierProb

        return classifications