import neuralnetwork as nn
import task

maxError = 0.02
numIterations = 50
count = 0
numInterventions = 0
averageList = []

for i in range(numInterventions+1):
    averageList.append(0)

t = task.Task(learningConstant=0.3, momentumConstant=0, inputNodes=1, hiddenNodes=1, outputNodes=1, auto=True, populationSize=1)

task.loadTask(t.task, preserveNet=False)
print(nn.learn(debug="full", errorCriterion=0.00001, strict=True, step=1, maxEpochs=1000, fullDebugStep=10000))
nn.testNetwork()
nn.getStrictError(debug=False)
# while count < numIterations:
#     print("Count", count)
#     t = task.Task(learningConstant=0.3, momentumConstant=0.5, inputNodes=5, hiddenNodes=, outputNodes=5, auto=True,
#                    populationSize=25)
#     interventions = []
#     for i in range(0, numInterventions):
#         interventions.append(t.popTask())
#
#     task.loadTask(t.task, preserveNet=False)
#
#     error = nn.learn(debug=False, errorCriterion=maxError)
#
#     if error <= maxError:
#         averageList.append(error)
#         for i, intervention in enumerate(interventions):
#             task.loadTask(intervention, preserveNet=True)
#             nn.learn(debug=False, errorCriterion=maxError)
#
#             t.pushTask(intervention)
#             task.loadTask(t.task, preserveNet=True)
#
#             result = nn.runEpoch(teach=False)
#             print(i+1, result)
#             averageList[i] += result
#         count += 1

# for i in range(0, len(averageList)):
#     averageList[i] = averageList[i] / numIterations
#
# print(averageList)

# print(nn.runEpoch(teach=False))
