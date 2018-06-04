import neuralnetwork as nn
from task import Task



num_epochs = 0
repeats = 500
for i in range(repeats):
    nn.initalise()
    learnCount = nn.learn(errorCriterion=0.0001)['numEpochs']
    num_epochs += learnCount
    print(i, learnCount)
num_epochs /= repeats

print("Num: ", num_epochs)

nn.testNetwork()
