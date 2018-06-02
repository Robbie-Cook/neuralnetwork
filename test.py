import neuralnetwork as nn
import task


print(task.Task(inputNodes=32, hiddenNodes=16, outputNodes=32, populationSize=20, auto=True).task)


nn.initalise()
for i in range(100):
    nn.runEpoch()
    # print(nn.getStrictError())

nn.testNetwork()
