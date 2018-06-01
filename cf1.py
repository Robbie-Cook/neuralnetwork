import neuralnetwork as nn
import tasks

def getStrictError():
    teacher = nn.teacher
    highestError = 0
    for rowNum in range(0, len(teacher)):
        nn.runNetwork(input=nn.inputPatterns[rowNum], teach=False, expectedOutput=nn.inputPatterns[rowNum])
        for i in range(0, len(teacher[rowNum])):
            error = abs(teacher[rowNum][i]-nn.nodes["output"][i])
            if error > highestError:
                highestError = error
    return highestError

maxError = 0.02
numIterations = 50
count = 0
numInterventions = 10
averageList = []

for i in range(numInterventions+1):
    averageList.append(0)

while count < numIterations:
    print("Count", count)
    t = tasks.Task(learningConstant=0.3, momentumConstant=0.5, inputNodes=32, hiddenNodes=16, outputNodes=32, auto=True,
                   populationSize=25)
    interventions = []
    for i in range(0, numInterventions):
        interventions.append(t.popTask())

    tasks.loadTask(t.task, preserveNet=False)

    error = nn.learn(debug=False, errorCriterion=maxError)

    if error <= maxError:
        averageList.append(error)
        for i, intervention in enumerate(interventions):
            tasks.loadTask(intervention, preserveNet=True)
            nn.learn(debug=False, errorCriterion=maxError)

            t.pushTask(intervention)
            tasks.loadTask(t.task, preserveNet=True)

            result = nn.runEpoch(teach=False)
            print(i+1, result)
            averageList[i] += result
        count += 1

for i in range(0, len(averageList)):
    averageList[i] = averageList[i] / numIterations

print(averageList)

# print(nn.runEpoch(teach=False))