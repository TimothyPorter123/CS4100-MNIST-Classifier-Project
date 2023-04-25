
import numpy
import keras
from keras.datasets import mnist
import BayesClassifier
import CNNClassifier
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)


def guess(classifications):
	return numpy.argmax(classifications)

def evaluate(model, verificationNumber):
	correctGuesses = 0
	for i in range(0, verificationNumber):
		prediction = guess(model.classify(x_test[i]))
		if(prediction == y_test[i]):
			#print("Correct! " + str(prediction))
			correctGuesses = correctGuesses + 1
		else:
			#print("Incorrect. Guessed: " + str(prediction) + " Actual: " + str(y_test[i]))
			pass
	print("Overall Accuracy: " + str(correctGuesses / verificationNumber))
	return correctGuesses / verificationNumber



trainingNumbers = [numpy.power(2, i) for i in range(0, 11)]
models = [(BayesClassifier.Bayes_Classifier().train((x_train[:i], y_train[:i]), 1), CNNClassifier.CNN_Classifier().train((x_train[:i], y_train[:i]), (x_test[:i], y_test[:i]))) for i in trainingNumbers]

accuracies = [[evaluate(models[i][o], 100) for i in range(0, 11)] for o in range(0, 2)]

plt.plot(trainingNumbers, accuracies[0], label='Naive Bayes')
plt.plot(trainingNumbers, accuracies[1], label='CNN')
plt.ylabel('Model Accuracy')
plt.xlabel('Training Quantity')
plt.show()