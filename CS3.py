import sys
import ast
import math
import numpy as np
from docplex.mp.model import Model
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralBiclustering
from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import KMeans
from sklearn.cluster import Birch
from sklearn.cluster import MiniBatchKMeans
import cplex
import KMedoids
import itertools


def calculateWaitingTime(path, nodesInSolution, service_costs, redsCoordinates, numberOfReds, distanceMatrix):
    firstNodeToExclude = path[1]

    for n in nodesInSolution:
        if n not in path:
            for i, j in path.items():
                if j == n:
                    secondNodeToExclude = i

    currentNode = 1
    timeUntilLastRed = 0
    while numberOfReds > 0:
        if currentNode == firstNodeToExclude or currentNode == secondNodeToExclude:
            # print(0.0)
            # print(tmp[currentNode][path[currentNode]])
            timeUntilLastRed += distanceMatrix[currentNode - 1][path[currentNode] - 1]
        else:
            # print(currentServiceTime[currentNode])
            # print(tmp[currentNode][path[currentNode]])
            timeUntilLastRed += distanceMatrix[currentNode - 1][path[currentNode] - 1] + service_costs[
                currentNode - 1]

        currentNode = path[currentNode]

        if currentNode in redsCoordinates.keys():
            numberOfReds -= 1

    return timeUntilLastRed


def interClusterMoveRed(tours, redNodes, waitingTimes, service_costs, distanceMatrix):
    redsInCluster = {}
    bestTour_i = None
    bestTour_j = None

    minReds = int(len(redNodes) / len(tours))
    maxReds = minReds + 1

    for t in range(0, len(tours)):
        redsInCluster[t] = []
        for (i, j) in tours[t]:
            if i in redNodes.keys():
                redsInCluster[t].append(i)

    min_ = math.inf
    for t1 in range(0, len(tours)):
        for t2 in range(0, len(tours)):
            if t1 != t2:
                if len(redsInCluster[t1]) == minReds and len(redsInCluster[t2]) == maxReds:
                    path_i = {}
                    previous_i = {}
                    path_j = {}
                    previous_j = {}

                    for (n1, n2) in tours[t1]:
                        path_i[n1] = n2
                        previous_i[n2] = n1

                    for (n1, n2) in tours[t2]:
                        path_j[n1] = n2
                        previous_j[n2] = n1

                    for red1 in redsInCluster[t1]:
                        for red2 in redsInCluster[t2]:
                            p_i = path_i.copy()
                            p_j = path_j.copy()

                            p_i[p_j[red2]] = red1
                            p_i[previous_i[red1]] = red2
                            p_i[red2] = p_j[red2]
                            p_j[previous_j[red2]] = p_j[p_j[red2]]
                            del p_j[p_j[red2]]
                            del p_j[red2]

                            nodesInSolution_t1 = []
                            for n1, n2 in p_i.items():
                                if n1 not in nodesInSolution_t1:
                                    nodesInSolution_t1.append(n1)
                                if n2 not in nodesInSolution_t1:
                                    nodesInSolution_t1.append(n2)

                            nodesInSolution_t2 = []
                            for n1, n2 in p_j.items():
                                if n1 not in nodesInSolution_t2:
                                    nodesInSolution_t2.append(n1)
                                if n2 not in nodesInSolution_t2:
                                    nodesInSolution_t2.append(n2)

                            newWaitingTime_i = calculateWaitingTime(p_i, nodesInSolution_t1, service_costs, redNodes,
                                                                    len(redsInCluster[t1]) + 1,
                                                                    distanceMatrix)

                            newWaitingTime_j = calculateWaitingTime(p_j, nodesInSolution_t2, service_costs, redNodes,
                                                                    len(redsInCluster[t2]) - 1,
                                                                    distanceMatrix)

                            if newWaitingTime_i + newWaitingTime_j < min_:
                                min_ = newWaitingTime_i + newWaitingTime_j
                                w_i = newWaitingTime_i
                                w_j = newWaitingTime_j
                                bestTour_i = p_i
                                bestTour_j = p_j
                                i_ = t1
                                j_ = t2

    if bestTour_i != None and bestTour_j != None:
        newTour_i = []

        for n1, n2 in bestTour_i.items():
            newTour_i.append((n1, n2))

        newTour_j = []
        for n1, n2 in bestTour_j.items():
            print(str(n1) + " " + str(n2))
            print(distanceMatrix[n1 - 1][n2 - 1])
            newTour_j.append((n1, n2))

        tours[i_] = newTour_i
        tours[j_] = newTour_j
        waitingTimes[i_] = w_i
        waitingTimes[j_] = w_j

    return tours, waitingTimes


def interClusterSwapReds(tours, redNodes, waitingTimes, service_costs, distanceMatrix):
    subsets = list(itertools.combinations(range(0, len(tours)), 2))

    min_ = math.inf
    for (i, j) in subsets:
        path_i = {}
        previous_i = {}
        path_j = {}
        previous_j = {}

        for (n1, n2) in tours[i]:
            path_i[n1] = n2
            previous_i[n2] = n1

        for (n1, n2) in tours[j]:
            path_j[n1] = n2
            previous_j[n2] = n1

        reds_i = []
        for n in path_i.keys():
            if n in redNodes.keys():
                reds_i.append(n)

        reds_j = []
        for n in path_j.keys():
            if n in redNodes.keys():
                reds_j.append(n)

        for red1 in reds_i:
            for red2 in reds_j:
                p_i = path_i.copy()
                p_j = path_j.copy()

                p_i[previous_i[red1]] = red2
                p_j[previous_j[red2]] = red1

                p_i[red2] = p_i[red1]
                del p_i[red1]

                p_j[red1] = p_j[red2]
                del p_j[red2]

                nodesInSolution_i = []
                for n1, n2 in tours[i]:
                    if n1 not in nodesInSolution_i:
                        nodesInSolution_i.append(n1)
                    if n2 not in nodesInSolution_i:
                        nodesInSolution_i.append(n2)

                nodesInSolution_j = []
                for n1, n2 in tours[j]:
                    if n1 not in nodesInSolution_j:
                        nodesInSolution_j.append(n1)
                    if n2 not in nodesInSolution_j:
                        nodesInSolution_j.append(n2)

                newWaitingTime_i = calculateWaitingTime(p_i, nodesInSolution_i, service_costs, redNodes, len(reds_i),
                                                        distanceMatrix)

                newWaitingTime_j = calculateWaitingTime(p_j, nodesInSolution_j, service_costs, redNodes, len(reds_j),
                                                        distanceMatrix)

                if newWaitingTime_i + newWaitingTime_j < min_:
                    min_ = newWaitingTime_i + newWaitingTime_j
                    w_i = newWaitingTime_i
                    w_j = newWaitingTime_j
                    bestTour_i = p_i
                    bestTour_j = p_j
                    i_ = i
                    j_ = j

    newTour_i = []
    for n1, n2 in bestTour_i.items():
        newTour_i.append((n1, n2))

    newTour_j = []
    for n1, n2 in bestTour_j.items():
        newTour_j.append((n1, n2))

    tours[i_] = newTour_i
    tours[j_] = newTour_j
    waitingTimes[i_] = w_i
    waitingTimes[j_] = w_j

    return tours, waitingTimes


def intraClusterSwapReds(tour, redNodes, greenNodes, waitingTime, service_costs, distanceMatrix):
    path = {}
    previous = {}
    bestTour = None

    for (i, j) in tour:
        path[i] = j
        previous[j] = i

    reds = []
    for n in path.keys():
        if n in redNodes.keys():
            reds.append(n)

    subsets = list(itertools.combinations(reds, 2))

    for swap in subsets:
        p = path.copy()

        p[previous[swap[0]]] = swap[1]
        p[previous[swap[1]]] = swap[0]

        p[swap[1]] = path[swap[0]]
        p[swap[0]] = path[swap[1]]

        nodesInSolution = []
        for n1, n2 in tour:
            if n1 not in nodesInSolution:
                nodesInSolution.append(n1)
            if n2 not in nodesInSolution:
                nodesInSolution.append(n2)

        newWaitingTime = calculateWaitingTime(p, nodesInSolution, service_costs, redNodes, len(reds), distanceMatrix)

        if waitingTime > newWaitingTime:
            waitingTime = newWaitingTime
            bestTour = p

    if bestTour is not None:
        newTour = []

        for i, j in bestTour.items():
            newTour.append((i, j))

    else:
        newTour = tour

    return newTour, waitingTime


def destroyTours(currentSolution, totalCosts, tourProfits, packingCosts, service_cost, profits,
                 travellingTimesMatrix, solutionValue):
    nodesDeleted = []

    for i in range(0, m):
        for k in range(0, (int(len(currentSolution[i][1:-1]) / 3) + (len(currentSolution[i][1:-1]) % 3 > 0))):
            tmp = math.inf

            for j in range(1, len(currentSolution[i]) - 1):
                node = currentSolution[i][j]
                previousNode = currentSolution[i][j - 1]
                nextNode = currentSolution[i][j + 1]
                ratio = profits[node] / (
                        service_cost[node] + travellingTimesMatrix[previousNode][node] + travellingTimesMatrix[node][
                    nextNode])

                if tmp > ratio:
                    tmp = ratio
                    nodeToDelete = node
                    prevNode = previousNode
                    nNode = nextNode
                    indexNode = j

            nodesDeleted.append((nodeToDelete, i))
            totalCosts[i] = totalCosts[i] + travellingTimesMatrix[prevNode][nNode] - (
                    service_cost[nodeToDelete] + travellingTimesMatrix[prevNode][nodeToDelete] +
                    travellingTimesMatrix[nodeToDelete][nNode])

            tourProfits[i] -= profits[nodeToDelete]
            currentSolution[i].pop(indexNode)
            packingCosts[i] -= service_cost[nodeToDelete]

    # print(currentSolution)
    # print("+++")
    return nodesDeleted


def modifyTour(tmax, tour, totalCost, profit, packingCost, travellingTimesMatrix, service_cost, profits, tabuTagOut,
               item, clusterIndex):
    tmp = math.inf
    tourModified = False

    for i in range(1, len(tour) - 1):
        if iteration > tabuTagOut[clusterIndex][tour[i]]:
            newTour = tour.copy()
            removedItem = newTour.pop(i)
            newProfit = profit - profits[removedItem] + profits[item]

            newTotalCost = totalCost - service_cost[removedItem] + service_cost[item]
            newTotalCost -= (travellingTimesMatrix[tour[i - 1]][removedItem] + travellingTimesMatrix[removedItem][
                tour[i + 1]])
            newTotalCost += (travellingTimesMatrix[tour[i - 1]][item] + travellingTimesMatrix[item][tour[i + 1]])

            if newTotalCost <= tmax:
                tourModified = True
                if tmp > newTotalCost:
                    deletedItem = removedItem
                    tmp = newTotalCost
                    modifiedTour = newTour
                    modifiedTour.insert(i, item)
                    profitTour = newProfit
                    newPackingCost = packingCost - service_cost[removedItem] + service_cost[item]
                    gap = profitTour - profit

    if tourModified:
        return modifiedTour, profitTour, deletedItem, tmp, newPackingCost, gap
    else:
        return None


def increaseTour(tmax, tour, totalCost, profit, travellingTimesMatrix, service_cost, profits, item):
    tmp = math.inf
    tourIncreased = False

    for i in range(0, len(tour) - 1):
        newTotalCost = totalCost
        newProfit = profit + profits[item]

        newTotalCost -= travellingTimesMatrix[tour[i]][tour[i + 1]]
        newTotalCost += (travellingTimesMatrix[tour[i]][item] + travellingTimesMatrix[item][tour[i + 1]])
        newTotalCost += service_cost[item]

        if newTotalCost <= tmax:
            tourIncreased = True
            if tmp > newTotalCost:
                tmp = newTotalCost
                newTour = tour.copy()
                newTour.insert(i + 1, item)
                profitTour = newProfit
                newPackingCost = packingCost + service_cost[item]
                gap = profitTour - profit

    if tourIncreased:
        return newTour, profitTour, tmp, newPackingCost, gap
    else:
        return None


def localSearch(clusters, tours, service_costs, greenNodes, redNodes, hospitalNodes, waitingTimes, distanceMatrix):
    print("IntraCluster")
    for i in range(0, len(tours)):
        tours[i], waitingTimes[i] = intraClusterSwapReds(tours[i], redNodes, greenNodes, waitingTimes[i],
                                                         service_costs, distanceMatrix)
        print(tours[i])
        print(waitingTimes[i])

    print("InterCluster")
    tours, waitingTimes = interClusterSwapReds(tours, redNodes, waitingTimes, service_costs, distanceMatrix)
    print(tours)
    print(waitingTimes)
    interClusterMoveRed(tours, redNodes, waitingTimes, service_costs, distanceMatrix)
    print(tours)
    print(waitingTimes)


def assignUniform(partitions, clusters, distanceMatrix):
    distances = {}
    for p in range(0, len(partitions)):
        distances[p] = {}
        distance = 0
        for c in range(0, len(clusters)):
            for n in partitions[c].keys():
                for n1 in clusters[c].keys():
                    distance += distanceMatrix[n][n1]

            distance /= (len(partitions[c]) * len(clusters[c]))
            distances[p][c] = distance

    min_ = math.inf
    allCombination = getAllCombination([], [], list(range(0, len(partitions))), list(range(0, len(partitions))))

    for combination in allCombination:
        value = 0
        for (i, j) in combination:
            value += distances[i][j]

        if min_ > value:
            min_ = value
            bestCombination = combination

    return bestCombination


def associateHospitals(hospitals, clusters, distanceMatrix):
    distances = {}
    for c in range(0, len(clusters)):
        distances[c] = {}
        distance = 0
        for clusterIndex in hospitals.keys():
            for h in hospitals[clusterIndex]:
                for n in clusters[c]:
                    distance += distanceMatrix[h][n]

                distance /= (len(hospitals[clusterIndex]) * len(clusters[c]))
                distances[c][clusterIndex] = distance

    min_ = math.inf
    allCombination = getAllCombination([], [], list(range(0, len(clusters))), list(range(0, len(clusters))))

    for combination in allCombination:
        value = 0
        for (i, j) in combination:
            value += distances[i][j]

        if min_ > value:
            min_ = value
            bestCombination = combination

    return bestCombination
    # hospitalsClustered = clusterizeData(algorithm, distanceHospitalMatrix, hospitalsCoordinates, len(clusters))
    # hospitalsInTeam = {}
    # teamsNeedHospitals = {}
    # teamsSurplusHospitals = {}
    # flag = True
    #
    # hospitalsKeys = list(hospitalsCoordinates.keys())
    # for t in range(0, len(hospitalsClustered)):
    #     for h in hospitalsClustered[t]:
    #         if t not in hospitalsInTeam:
    #             hospitalsInTeam[t] = [h]
    #         else:
    #             hospitalsInTeam[t] += [h]
    #
    # maxRedsInTeam = 0
    # for t in redsInTeam:
    #     if maxRedsInTeam < redsInTeam[t]:
    #         maxRedsInTeam = redsInTeam[t]
    #
    # for t in redsInTeam.keys():
    #     if len(hospitalsInTeam[t]) <= maxRedsInTeam:
    #         teamsNeedHospitals[t] = maxRedsInTeam - len(hospitalsInTeam[t]) + 1
    #     else:
    #         teamsSurplusHospitals[t] = len(hospitalsInTeam[t]) - maxRedsInTeam - 1
    #
    # while flag:
    #     c = 0
    #     for t in teamsNeedHospitals:
    #         c += teamsNeedHospitals[t]
    #
    #     if c > 0:
    #         sortedDict = dict(sorted(teamsSurplusHospitals.items(), key=lambda item: item[1], reverse=True))
    #         for t in teamsNeedHospitals:
    #             min_ = math.inf
    #             distance = 0
    #             for h in hospitalsInTeam[list(sortedDict.keys())[0]]:
    #                 for n in clusters[t]:
    #                     distance += distanceMatrix[n - 1][h - 1]
    #
    #             if min_ > distance:
    #                 min_ = distance
    #                 swap = (h, list(sortedDict.keys())[0], t)
    #
    #         teamsNeedHospitals[swap[2]] -= 1
    #         teamsSurplusHospitals[swap[1]] -= 1
    #         hospitalsInTeam[swap[2]] += [swap[0]]
    #         hospitalsInTeam[swap[1]].remove(swap[0])
    #     else:
    #         flag = False
    #
    # distances = {}
    #
    # for c in range(0, len(clusters)):
    #     distances[c] = {}
    #     distance = 0
    #     for t in hospitalsInTeam.keys():
    #         for n in clusters[c]:
    #             for h in hospitalsInTeam[t]:
    #                 distance += distanceMatrix[h][n]
    #
    #         distance /= (len(clusters[c]) * len(hospitalsInTeam[t]))
    #         distances[c][t] = distance
    #
    # min_ = math.inf
    # allCombination = getAllCombination([], [], list(range(0, len(clusters))), list(range(0, len(clusters))))
    #
    # for combination in allCombination:
    #     value = 0
    #     for (i, j) in combination:
    #         value += distances[i][j]
    #
    #     if min_ > value:
    #         min_ = value
    #         bestCombination = combination
    #
    # return bestCombination, hospitalsInTeam


def getAllCombination(allCombination, currentCombination, firstList, secondList):
    if len(firstList) == 1 and len(secondList) == 1:
        combination = currentCombination + [(firstList[0], secondList[0])]
        allCombination.append(combination)
        return allCombination
    else:
        for i in range(0, len(firstList)):
            sl = secondList.copy()
            tmp = [(firstList[0], sl.pop(i))]
            allCombination = getAllCombination(allCombination, currentCombination + tmp, firstList[1:], sl)

        return allCombination


def associateRedsGreens(reds, clusters, distanceMatrix):
    distances = {}
    for c in range(0, len(clusters)):
        distances[c] = {}
        distance = 0
        for clusterIndex in reds.keys():
            for red in reds[clusterIndex]:
                for green in clusters[c]:
                    distance += distanceMatrix[red][green]

                distance /= (len(reds[clusterIndex]) * len(clusters[c]))
                distances[c][clusterIndex] = distance

    min_ = math.inf
    allCombination = getAllCombination([], [], list(range(0, len(clusters))), list(range(0, len(clusters))))

    for combination in allCombination:
        value = 0
        for (i, j) in combination:
            value += distances[i][j]

        if min_ > value:
            min_ = value
            bestCombination = combination

    return bestCombination


def assignUniformNodes(partitions, coordinates, m, partialDistanceMatrix, completeDistanceMatrix, partitionElements):
    flag = True
    mapPartitionsClusters = {}
    clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)

    if partitionElements > 0:
        bestCombination = assignUniform(partitions, clusters, completeDistanceMatrix)
        print(bestCombination)

        for (p, c) in bestCombination:
            mapPartitionsClusters[p] = c
            partitions[p] = {**clusters[c], **partitions[p]}
    else:
        for i in range(0, m):
            mapPartitionsClusters[i] = i
            partitions[i] = {**clusters[i], **partitions[i]}

    minNodes = int((partitionElements + len(coordinates)) / m)
    maxNodes = minNodes + 1

    while flag:
        flag = False
        surplusClusters = []
        lackClusters = []

        for c in range(0, len(partitions)):
            if len(partitions[c]) >= maxNodes:
                surplusClusters.append(c)
            if len(partitions[c]) < minNodes:
                lackClusters.append(c)

            if len(partitions[c]) > maxNodes or len(partitions[c]) < minNodes:
                flag = True

        if flag:
            if len(lackClusters) == 0:
                lackClusters = range(0, len(partitions))
                lackClusters = list(set(lackClusters) - set(surplusClusters))

            min_ = math.inf

            for c in surplusClusters:
                for c1 in lackClusters:
                    for n in partitions[c]:
                        if n in clusters[mapPartitionsClusters[c]]:
                            distance = 0
                            for n1 in partitions[c1]:
                                distance += completeDistanceMatrix[n - 1][n1 - 1]

                            distance /= len(partitions[c1])

                            if min_ > distance:
                                min_ = distance
                                swap = (n, c, c1)

            partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
            del partitions[swap[1]][swap[0]]

    return partitions

    # for i in range(0, m):
    #     clusters.append({})
    #     nodesInCluster[i] = 0
    #
    # maxNodes = int(len(coordinates) / m)
    #
    # for node in coordinates.keys():
    #     k = 0
    #     min_ = math.inf
    #
    #     for c in range(0, len(clusters)):
    #         if nodesInCluster[c] < maxNodes:
    #             distance = 0
    #             for n in clusters[c]:
    #                 distance += distanceMatrix[node - 1][n - 1]
    #
    #             if distance < min_:
    #                 min_ = distance
    #                 toCluster = c
    #         else:
    #             k += 1
    #
    #     if k == m:
    #         for c in range(0, len(clusters)):
    #             nodesInCluster[c] -= 1
    #
    #         for c in range(0, len(clusters)):
    #             distance = 0
    #             for n in clusters[c]:
    #                 distance += distanceMatrix[node - 1][n - 1]
    #
    #             if distance < min_:
    #                 min_ = distance
    #                 toCluster = c
    #
    #     clusters[toCluster][node] = coordinates[node]
    #     nodesInCluster[c] += 1
    #
    # return clusters

    # mdl = Model(name='uniformGreens')
    # xVars = []
    # yVars = []
    #
    # for j in range(0, greenNodes):
    #     vars = []
    #     for i in range(0, m):
    #         vars.append(mdl.integer_var(lb=0.0, ub=1.0, name='x' + str(j) + "_" + str(i)))
    #     xVars.append(vars)
    #
    # for i in range(0, m):
    #     yVars.append(mdl.integer_var(lb=0.0, ub=greenNodes, name='y' + str(i)))
    #
    # for j in range(0, greenNodes):
    #     mdl.add_constraint(mdl.sum(xVars[j][i] for i in range(0, m)) == 1)
    #
    # for i in range(0, m):
    #     mdl.add_constraint(mdl.sum(xVars[j][i] for j in range(0, greenNodes)) == yVars[i])
    #
    # mdl.minimize(mdl.sum(mdl.abs(yVars[i] - yVars[j]) for i in range(0, m) for j in range(i + 1, m)))
    # mdl.export("assignUniformGreens.lp")
    # sol = mdl.solve()
    #
    # if sol is not None:
    #     assignment = []
    #     for v in mdl.iter_integer_vars():
    #         if v.solution_value > 0 and "x" in v.to_string():
    #             var = v.to_string()
    #             node = int(var[1:var.index("_")])
    #             var = var[var.index("_") + 1:len(var)]
    #             team = int(var[0:len(var)])
    #             assignment.append((node, team))
    # else:
    #     return None
    #
    # return assignment


def assignUniformReds(redNodes, m, hospitals):
    mdl = Model(name='uniformReds')
    xVars = []
    yVars = []

    for j in range(0, redNodes):
        vars = []
        for i in range(0, m):
            vars.append(mdl.integer_var(lb=0.0, ub=1.0, name='x' + str(j) + "_" + str(i)))
        xVars.append(vars)

    for i in range(0, m):
        yVars.append(mdl.integer_var(lb=0.0, ub=redNodes, name='y' + str(i)))

    for i in range(0, m):
        mdl.add_constraint(mdl.sum(xVars[j][i] for j in range(0, redNodes)) == yVars[i])

    mdl.add_constraint(mdl.sum(yVars[i] + 1 for i in range(0, m)) <= hospitals)

    for j in range(0, redNodes):
        mdl.add_constraint(mdl.sum(xVars[j][i] for i in range(0, m)) == 1)

    mdl.minimize(mdl.sum(mdl.abs(yVars[i] - yVars[j]) for i in range(0, m) for j in range(i + 1, m)))
    mdl.export("assignUniformReds.lp")
    sol = mdl.solve()

    if sol is not None:
        assignment = []
        for v in mdl.iter_integer_vars():
            if v.solution_value > 0 and "x" in v.to_string():
                var = v.to_string()
                node = int(var[1:var.index("_")])
                var = var[var.index("_") + 1:len(var)]
                team = int(var[0:len(var)])
                assignment.append((node, team))
    else:
        return None

    return assignment


def assignUniformHospitals(partitions, coordinates, m, partialDistanceMatrix, completeDistanceMatrix, redNodes,
                           partitionElements):
    mapPartitionsClusters = {}
    gapRedsHospitalsInCluster = {}
    minHospitalsInCluster = {}
    maxHospitalsInCluster = {}
    redsInCluster = {}
    flag = True

    clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
    bestCombination = assignUniform(partitions, clusters, completeDistanceMatrix)
    print(bestCombination)

    for (p, c) in bestCombination:
        mapPartitionsClusters[p] = c
        partitions[p] = {**clusters[c], **partitions[p]}

    for c in range(0, len(partitions)):
        for r in partitions[c].keys():
            if r in redNodes.keys():
                if c not in redsInCluster:
                    redsInCluster[c] = 1
                else:
                    redsInCluster[c] += 1

        minHospitalsInCluster[c] = redsInCluster[c] + 1
        maxHospitalsInCluster[c] = minHospitalsInCluster[c] + 1

    minNodes = int((partitionElements + len(coordinates)) / m)
    maxNodes = minNodes + 1

    while flag:
        flag = False
        surplusClusters = []
        lackClusters = []
        hospitalsInClusters = {}

        for c in range(0, len(partitions)):
            for h in partitions[c].keys():
                if h in coordinates.keys():
                    if c not in hospitalsInClusters:
                        hospitalsInClusters[c] = 1
                    else:
                        hospitalsInClusters[c] += 1

            if hospitalsInClusters[c] >= maxHospitalsInCluster[c]:
                surplusClusters.append(c)
            if hospitalsInClusters[c] < minHospitalsInCluster[c]:
                lackClusters.append(c)

            if hospitalsInClusters[c] < minHospitalsInCluster[c]:
                flag = True

        if flag:
            min_ = math.inf

            for c in surplusClusters:
                for c1 in lackClusters:
                    for n in partitions[c]:
                        if n in clusters[mapPartitionsClusters[c]]:
                            distance = 0
                            for n1 in partitions[c1]:
                                distance += completeDistanceMatrix[n - 1][n1 - 1]

                            distance /= len(partitions[c1])

                            if min_ > distance:
                                min_ = distance
                                swap = (n, c, c1)

            partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
            del partitions[swap[1]][swap[0]]
        else:
            surplusClusters = []
            lackClusters = []

            for c in range(0, len(partitions)):
                if len(partitions[c]) >= maxNodes and hospitalsInClusters[c] > minHospitalsInCluster[c]:
                    surplusClusters.append(c)
                if len(partitions[c]) < minNodes:
                    lackClusters.append(c)

                if len(partitions[c]) > maxNodes or len(partitions[c]) < minNodes:
                    flag = True

            if flag:
                if len(lackClusters) == 0:
                    lackClusters = range(0, len(partitions))
                    lackClusters = list(set(lackClusters) - set(surplusClusters))

                min_ = math.inf

                for c in surplusClusters:
                    for c1 in lackClusters:
                        for n in partitions[c]:
                            if n in clusters[mapPartitionsClusters[c]]:
                                distance = 0
                                for n1 in partitions[c1]:
                                    distance += completeDistanceMatrix[n - 1][n1 - 1]

                                distance /= len(partitions[c1])

                                if min_ > distance:
                                    min_ = distance
                                    swap = (n, c, c1)

                partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
                del partitions[swap[1]][swap[0]]

    return partitions
    # clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
    # gapRedsHospitalsInCluster = {}
    # gapInCluster = {}
    # redsInCluster = {}
    # flag = True
    #
    # maxNodes = int((partitionElements + len(coordinates)) / m)
    #
    # for c in range(0, len(partitions)):
    #     for r in partitions[c].keys():
    #         if r in redNodes.keys():
    #             if c not in redsInCluster:
    #                 redsInCluster[c] = 1
    #             else:
    #                 redsInCluster[c] += 1
    #
    # for i in range(0, m):
    #     partitions[i] = {**clusters[i], **partitions[i]}
    #
    # while flag:
    #     flag = False
    #     surplusClusters = []
    #     lackClusters = []
    #     hospitalsInClusters = {}
    #
    #     for c in range(0, len(partitions)):
    #         for h in partitions[c].keys():
    #             if h in coordinates.keys():
    #                 if c not in hospitalsInClusters:
    #                     hospitalsInClusters[c] = 1
    #                 else:
    #                     hospitalsInClusters[c] += 1
    #
    #         gapRedsHospitalsInCluster[c] = hospitalsInClusters[c] - redsInCluster[c] - 1
    #
    #         if gapRedsHospitalsInCluster[c] > 1:
    #             surplusClusters.append(c)
    #         elif gapRedsHospitalsInCluster[c] < 1:
    #             lackClusters.append(c)
    #
    #         if gapRedsHospitalsInCluster[c] < 1:
    #             flag = True
    #
    #     if flag:
    #         min_ = math.inf
    #
    #         for c in surplusClusters:
    #             for c1 in lackClusters:
    #                 for n in partitions[c]:
    #                     if n in clusters[c]:
    #                         distance = 0
    #                         for n1 in partitions[c1]:
    #                             distance += completeDistanceMatrix[n - 1][n1 - 1]
    #
    #                         distance /= len(partitions[c1])
    #
    #                         if min_ > distance:
    #                             min_ = distance
    #                             swap = (n, c, c1)
    #
    #         partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
    #         del partitions[swap[1]][swap[0]]
    #     else:
    #         surplusClusters_ = []
    #         lackClusters_ = []
    #
    #         for c in surplusClusters:
    #             gapInCluster[c] = len(partitions[c]) - maxNodes
    #
    #             if gapInCluster[c] != 0:
    #                 if gapInCluster[c] > 0:
    #                     if c not in surplusClusters_:
    #                         surplusClusters_.append(c)
    #                 elif gapInCluster[c] < 0:
    #                     if c not in lackClusters_:
    #                         lackClusters_.append(c)
    #
    #             if len(partitions[c]) < maxNodes:
    #                 flag = True
    #
    #         if flag:
    #             min_ = math.inf
    #
    #             for c in surplusClusters_:
    #                 for c1 in lackClusters_:
    #                     for n in partitions[c]:
    #                         if n in clusters[c]:
    #                             distance = 0
    #                             for n1 in partitions[c1]:
    #                                 distance += completeDistanceMatrix[n - 1][n1 - 1]
    #
    #                             distance /= len(partitions[c1])
    #
    #                             if min_ > distance:
    #                                 min_ = distance
    #                                 swap = (n, c, c1)
    #
    #             partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
    #             del partitions[swap[1]][swap[0]]
    #
    # return partitions
    # mdl = Model(name='uniformHospitals')
    # xVars = []
    # yVars = []
    #
    # for j in range(0, hospitals):
    #     vars = []
    #     for i in range(0, m):
    #         vars.append(mdl.integer_var(lb=0.0, ub=1.0, name='x' + str(j) + "_" + str(i)))
    #     xVars.append(vars)
    #
    # for i in range(0, m):
    #     yVars.append(mdl.integer_var(lb=0.0, ub=hospitals, name='y' + str(i)))
    #
    # for j in range(0, hospitals):
    #     mdl.add_constraint(mdl.sum(xVars[j][i] for i in range(0, m)) == 1)
    #
    # for i in range(0, m):
    #     mdl.add_constraint(mdl.sum(xVars[j][i] for j in range(0, hospitals)) == yVars[i])
    #
    # mdl.minimize(mdl.sum(mdl.abs(yVars[i] - yVars[j]) for i in range(0, m) for j in range(i + 1, m)))
    # mdl.export("assignUniformHospitals.lp")
    # sol = mdl.solve()
    #
    # if sol is not None:
    #     assignment = []
    #     for v in mdl.iter_integer_vars():
    #         if v.solution_value > 0 and "x" in v.to_string():
    #             var = v.to_string()
    #             node = int(var[1:var.index("_")])
    #             var = var[var.index("_") + 1:len(var)]
    #             team = int(var[0:len(var)])
    #             assignment.append((node, team))
    # else:
    #     return None
    #
    # return assignment


# def swapNodeBetweenCliusters(clusters, clustersLack, clustersSurplus, distanceMatrix_, nodes):
#     min_ = math.inf
#     fromCluster = None
#     toCluster = None
#     node = None
#
#     for cs in clustersSurplus:
#         for cl in clustersLack:
#             meanDistance = 0
#             for n1 in list(set(clusters[cs]) & set(nodes)):
#                 for n2 in clusters[cl]:
#                     meanDistance += distanceMatrix_[n1 - 2][n2 - 2]
#
#                 meanDistance /= len(clusters[cl])
#
#                 if meanDistance < min_:
#                     min_ = meanDistance
#                     fromCluster = cs
#                     toCluster = cl
#                     node = n1
#
#     clusters[toCluster][node] = clusters[fromCluster][node]
#     del clusters[fromCluster][node]
#
#     return node, fromCluster, toCluster
#
#
# def calculateLackSurplus(clusters, nodes, m, greens, reds, hospitals):
#     greensLack = []
#     greensSurplus = []
#     redsLack = []
#     redsSurplus = []
#     hospitalsLack = []
#     hospitalsSurplus = []
#
#     for c in range(0, len(clusters)):
#         greensInCluster = 0
#         redsInCluster = 0
#         hospitalsInCluster = 0
#
#         for n in clusters[c]:
#             if n in greens:
#                 greensInCluster += 1
#         for n in clusters[c]:
#             if n in reds:
#                 redsInCluster += 1
#         for n in clusters[c]:
#             if n in hospitals:
#                 hospitalsInCluster += 1
#
#         if greensInCluster < int((len(greens) / m)):
#             greensLack.append(c)
#
#         if redsInCluster < int((len(reds) / m)):
#             redsLack.append(c)
#
#         if hospitalsInCluster < int((len(hospitals) / m)):
#             hospitalsLack.append(c)
#
#         if greensInCluster > int((len(greens) / m)):
#             greensSurplus.append(c)
#
#         if redsInCluster > int((len(reds) / m)):
#             redsSurplus.append(c)
#
#         if hospitalsInCluster > int((len(hospitals) / m)):
#             hospitalsSurplus.append(c)
#
#     return greensLack, greensSurplus, redsLack, redsSurplus, hospitalsLack, hospitalsSurplus


def clusterizeData(algorithm, distanceMatrix, coordinates, m):
    clusters = []

    if algorithm == 0:
        clustering = AgglomerativeClustering(n_clusters=m, linkage="complete").fit(list(coordinates.values()))
    elif algorithm == 1:
        clustering = KMeans(n_clusters=m, random_state=0).fit(list(coordinates.values()))
    elif algorithm == 2:
        clustering = KMedoids.KMedoids(n_clusters=m, random_state=0, init="k-medoids++").fit(list(coordinates.values()))
    elif algorithm == 3:
        clustering = MiniBatchKMeans(n_clusters=m, random_state=0, batch_size=5).fit(list(coordinates.values()))
    elif algorithm == 4:
        clustering = SpectralClustering(
            n_clusters=m, assign_labels="discretize", random_state=0, affinity='precomputed').fit(distanceMatrix)

    for i in range(0, m):
        clusters.append({})

    c = 0
    for i in coordinates.keys():
        clusters[clustering.labels_[c]][i] = coordinates[i]
        c += 1

    return clusters


def getFeasibleTour(distanceMatrix, service_costs, h, r, g, scores, nodes, nReds, tmax):
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)

    names = []
    upper_bounds = []
    lower_bounds = []
    types = []
    constraint_names = []
    constraints = []

    greensHospitalsMatrix = []
    for i in range(0, nodes):
        v = []
        for j in range(0, nodes):
            v.append(g[i] * h[j])
        greensHospitalsMatrix.append(v)

    redsGreensMatrix = []
    for i in range(0, nodes):
        v = []
        for j in range(0, nodes):
            v.append(r[i] * g[j])
        redsGreensMatrix.append(v)

    hospitals = h
    reds = r
    greens = g

    minusReds = []
    for i in range(1, nodes - 1):
        minusReds.append(0.95 * -reds[i])

    # minusDistanceTimes = []
    # for i in range(0, nodes):
    #     for j in range(0, nodes):
    #         if i != j:
    #             minusDistanceTimes.append(0.50 * -distanceMatrix[i][j])

    for i in range(0, nodes):
        names.append("y" + str(i))
        types.append(problem.variables.type.integer)
        upper_bounds.append(1.0)
        lower_bounds.append(0.0)

    for i in range(1, nodes - 1):
        names.append("u" + str(i))
        types.append(problem.variables.type.integer)
        upper_bounds.append(nodes - 2)
        lower_bounds.append(0.0)

    for i in range(0, nodes):
        for j in range(0, nodes):
            if i != j:
                names.append("x" + str(i) + "_" + str(j))
                types.append(problem.variables.type.integer)
                upper_bounds.append(1.0)
                lower_bounds.append(0.0)

    names.append("d")
    types.append(problem.variables.type.continuous)
    upper_bounds.append(cplex.infinity)
    lower_bounds.append(0.0)

    objective = [s * 0.05 for s in scores] + minusReds + [0.0] * (nodes ** 2 - nodes) + [0.0]

    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          types=types,
                          names=names)

    # Constraints
    constraintsNumber = ((nodes - 1) * 2) + ((nodes - 2) ** 2 - (nodes - 2)) + 10
    for i in range(0, constraintsNumber):
        constraint_names.append("c" + str(i))

    # Da ogni nodo j può entrare al più un arco (se il nodo j è in soluzione)
    for j in range(1, nodes):
        variables = []
        for i in range(0, nodes):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
        variables.append("y" + str(j))
        constraints.append([variables, ([1.0] * (nodes - 1)) + [-1.0]])

    # Da ogni nodo j può uscire al più un arco (se il nodo j è in soluzione)
    for i in range(0, nodes - 1):
        variables = []
        for j in range(0, nodes):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
        variables.append("y" + str(i))
        constraints.append([variables, ([1.0] * (nodes - 1)) + [-1.0]])

    # Miller–Tucker–Zemlin subtour elimination
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                constraints.append(
                    [["u" + str(i), "u" + str(j), "x" + str(i) + "_" + str(j)], [1.0, -1.0, nodes - 1]])

    # Non possono esistere archi del tipo: r -> g
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
                coefficients.append(redsGreensMatrix[i][j])
    constraints.append([variables, coefficients])

    # Non possono esistere archi del tipo: h -> h
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
                coefficients.append(hospitals[i] * hospitals[j])
    constraints.append([variables, coefficients])

    # # Non possono esistere archi del tipo: g -> h
    # variables = []
    # coefficients = []
    # for i in range(1, nodes - 1):
    #     for j in range(1, nodes - 1):
    #         if i != j:
    #             variables.append("x" + str(i) + "_" + str(j))
    #             coefficients.append(greensHospitalsMatrix[i][j])
    # constraints.append([variables, coefficients])

    # Non possono esistere archi del tipo: r -> r
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
                coefficients.append(reds[i] * reds[j])
    constraints.append([variables, coefficients])

    # variables = []
    # coefficients = []
    # for i in range(1, nodes - 1):
    #     for j in range(1, nodes - 1):
    #         if i != j:
    #             variables.append("x" + str(i) + "_" + str(j))
    #             coefficients.append(reds[i] * hospitals[j])
    # constraints.append([variables, coefficients])
    #
    # variables = []
    # coefficients = []
    # for i in range(1, nodes - 1):
    #     for j in range(1, nodes - 1):
    #         if i != j:
    #             variables.append("x" + str(i) + "_" + str(j))
    #             coefficients.append(greens[i] * reds[j])
    # constraints.append([variables, coefficients])
    #
    # variables = []
    # coefficients = []
    # for i in range(1, nodes - 1):
    #     for j in range(1, nodes - 1):
    #         if i != j:
    #             variables.append("x" + str(i) + "_" + str(j))
    #             coefficients.append(greens[i] * hospitals[j])
    # constraints.append([variables, coefficients])
    #
    # variables = []
    # coefficients = []
    # for i in range(1, nodes - 1):
    #     for j in range(1, nodes - 1):
    #         if i != j:
    #             variables.append("x" + str(i) + "_" + str(j))
    #             coefficients.append(hospitals[i] * greens[j])
    # constraints.append([variables, coefficients])

    # Il nodo di partenza deve essere un h
    variables = []
    coefficients = []
    for i in range(1, nodes):
        variables.append("x0" + "_" + str(i))
        coefficients.append(hospitals[i])
    constraints.append([variables, coefficients])

    # Il nodo di arrivo deve essere un h
    variables = []
    coefficients = []
    for i in range(0, nodes - 1):
        variables.append("x" + str(i) + "_" + str(nodes - 1))
        coefficients.append(hospitals[i])
    constraints.append([variables, coefficients])

    # Nessun arco deve entrare in 0
    variables = []
    coefficients = []
    for i in range(1, nodes):
        variables.append("x" + str(i) + "_0")
        coefficients.append(1.0)
    constraints.append([variables, coefficients])

    # Nessun arco deve uscire da N
    variables = []
    coefficients = []
    for i in range(0, nodes - 1):
        variables.append("x" + str(nodes - 1) + "_" + str(i))
        coefficients.append(1.0)
    constraints.append([variables, coefficients])

    # I rossi devo essere visitati tutti
    variables = []
    coefficients = []
    for i in range(0, nodes):
        variables.append("y" + str(i))
        coefficients.append(reds[i])
    constraints.append([variables, coefficients])

    # definisco d come somma dei tempi di servizio del primo e ultimo ospedale
    variables = []
    coefficients = []
    for i in range(1, nodes):
        variables.append("x0" + "_" + str(i))
        coefficients.append(service_costs[i])

    for i in range(1, nodes - 1):
        variables.append("x" + str(i) + "_" + str(nodes - 1))
        coefficients.append(service_costs[i])

    variables.append("d")
    coefficients.append(-1.0)
    constraints.append([variables, coefficients])

    # il costo del tour non deve eccedere tmax + d
    variables = []
    coefficients = []
    for i in range(0, nodes):
        variables.append("y" + str(i))
        coefficients.append(service_costs[i])
        for j in range(0, nodes):
            if i != j:
                variables.append("x" + str(i) + "_" + str(j))
                coefficients.append(distanceMatrix[i][j])

    variables.append("d")
    coefficients.append(-1.0)
    constraints.append([variables, coefficients])

    rhs = ([0.0] * ((nodes - 1) * 2)) + ([(nodes - 2)] * ((nodes - 2) ** 2 - (nodes - 2))) + [0.0] + [0.0] + [
        0.0] + ([1.0] * 2) + ([0.0] * 2) + [nReds] + [0.0] + [tmax]

    constraint_senses = (["E"] * ((nodes - 1) * 2)) + (["L"] * ((nodes - 2) ** 2 - (nodes - 2))) + ["E"] + [
        "E"] + ["E"] + (["E"] * 2) + (["E"] * 2) + ["E"] + ["E"] + ["L"]

    problem.linear_constraints.add(lin_expr=constraints, senses=constraint_senses, rhs=rhs, names=constraint_names)
    problem.write("prob.lp")

    # Solve the problem
    problem.solve()

    # And print the solutions
    print(problem.solution.get_objective_value())
    variables = problem.variables.get_names()
    values = problem.solution.get_values()

    for i in range(0, len(variables)):
        if ("u" in variables[i]):
            print(variables[i] + " = " + str(values[i]))

    solution = []
    for i in range(0, len(variables)):
        if (values[i] == 1.0 and "x" in variables[i]):
            v = variables[i][1:len(variables[i])]
            firstNode = v[0: v.index('_')]
            secondNode = v[v.index('_') + 1: len(v)]
            solution.append((int(firstNode), int(secondNode)))

    y = []
    for i in range(0, len(variables)):
        if (values[i] == 1.0 and "y" in variables[i]):
            v = variables[i][1:len(variables[i])]
            y.append(int(v))

    return (problem.solution.get_objective_value(), solution, y)


f = open(str(sys.argv[1]), "r")
f1 = open(str(sys.argv[2]), "r")

parameters = []
clusters = []
distanceMatrix = []
tours = []
waitingTimes = []
solutionNodes = []

for i in range(0, 10):
    s = f.readline()
    s = s[s.index('=') + 1: len(s)]
    parameters.append(s.rstrip("\n"))

parameters[0] = int(parameters[0])
print(parameters[0])
parameters[1] = int(parameters[1])
print(parameters[1])
parameters[2] = list(ast.literal_eval(parameters[2]))
print(parameters[2])
parameters[3] = list(ast.literal_eval(parameters[3]))
print(parameters[3])
parameters[4] = list(ast.literal_eval(parameters[4]))
print(parameters[4])
parameters[5] = list(ast.literal_eval(parameters[5]))
print(parameters[5])
parameters[6] = list(ast.literal_eval(parameters[6]))
print(parameters[6])
parameters[7] = ast.literal_eval(parameters[7])
print(parameters[7])
parameters[8] = int(parameters[8])
print(parameters[8])
parameters[9] = ast.literal_eval(parameters[9])
print(parameters[9])

for i in range(0, parameters[1]):
    clusters.append({})

s = f.readline()

i = 0
while True:
    s = f.readline()
    if s != "":
        if i == 0:
            distanceMatrix.append(ast.literal_eval(s[1:-2]))
        elif i == (parameters[0] - 1):
            distanceMatrix.append(ast.literal_eval(s[0:-1]))
        else:
            distanceMatrix.append(ast.literal_eval(s[0:-2]))

        i += 1
    else:
        break

for i in range(9):
    f1.readline()

coordinates = {}
greensCoordinates = {}
redsCoordinates = {}
hospitalsCoordinates = {}

for i in range(1, parameters[0] - 1):
    line = f1.readline()
    coords = line.split()
    coordinates[int(coords[0])] = [float(coords[1]), float(coords[2])]

for key in coordinates.keys():
    if key in parameters[3]:
        greensCoordinates[key] = coordinates[key]

for key in coordinates.keys():
    if key in parameters[4]:
        redsCoordinates[key] = coordinates[key]

for key in coordinates.keys():
    if key in parameters[5]:
        hospitalsCoordinates[key] = coordinates[key]

# distanceMatrix_ = np.array(distanceMatrix, dtype=float)
# distanceMatrix_ = np.delete(distanceMatrix_, parameters[0] - 1, 0)
# distanceMatrix_ = np.delete(distanceMatrix_, parameters[0] - 1, 1)
# distanceMatrix_ = np.delete(distanceMatrix_, 0, 0)
# distanceMatrix_ = np.delete(distanceMatrix_, 0, 1)

# nodesDeleted = 0
# for p in range(parameters[6][1] - 1, -1, -1):
#     if (p + 1) not in parameters[3]:
#         distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
#         distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
#         # nodesDeleted += 1

# clusters = assignUniformGreens(greensCoordinates, parameters[1], distanceMatrix)

# for (i, j) in assignment:
#     keys = list(greensCoordinates.keys())
#     clusters[j][keys[i]] = greensCoordinates[keys[i]]

# print(clusters)

# clusters = clusterizeData(int(sys.argv[3]), distanceMatrix_, greenPatients, parameters[1])
# print(clusters)
# greensLack, greensSurplus, redsLack, redsSurplus, hospitalsLack, hospitalsSurplus \
#     = calculateLackSurplus(clusters, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5])
#
# while len(greensLack) > 0 or len(redsLack) > 0 or len(hospitalsLack) > 0:
#     if len(greensLack) > 0:
#         print(swapNodeBetweenCliusters(clusters, greensLack, greensSurplus, distanceMatrix_, parameters[3]))
#     if len(redsLack) > 0:
#         print(swapNodeBetweenCliusters(clusters, redsLack, redsSurplus, distanceMatrix_, parameters[4]))
#     if len(hospitalsLack) > 0:
#         print(swapNodeBetweenCliusters(clusters, hospitalsLack, hospitalsSurplus, distanceMatrix_, parameters[5]))
#
#     greensLack, greensSurplus, redsLack, redsSurplus, hospitalsLack, hospitalsSurplus \
#         = calculateLackSurplus(clusters, parameters[0], parameters[1], parameters[3], parameters[4], parameters[5])

# weights = []
# for p in parameters[4]:
#     weights.append(parameters[7][p - 1])

distanceMatrix_ = np.array(distanceMatrix, dtype=float)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[3]:
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

clusters = assignUniformNodes(clusters, greensCoordinates, parameters[1], distanceMatrix_, distanceMatrix, 0)

# if assignment is None:
#     sys.exit("Non vi è almeno un ospedale per team")

# reds = {}
# for (red, cluster) in assignment:
#     if cluster not in reds:
#         reds[cluster] = [parameters[4][red]]
#     else:
#         reds[cluster].append(parameters[4][red])
#
# partitions = associateRedsGreens(reds, clusters, distanceMatrix)

distanceMatrix_ = np.array(distanceMatrix, dtype=float)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[4]:
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

clusters = assignUniformNodes(clusters, redsCoordinates, parameters[1], distanceMatrix_, distanceMatrix,
                              len(greensCoordinates))

# redsInTeam = {}
# for (i, j) in partitions:
#     for n in reds[i]:
#         clusters[i][n] = coordinates[n]
# if i not in redsInTeam:
#     redsInTeam[i] = 1
# else:
#     redsInTeam[i] += 1


# hospitalsCoordinates = {}
# for h in parameters[5]:
#     hospitalsCoordinates[h] = coordinates[h]

# distanceHospitalsMatrix = np.array(distanceMatrix, dtype=float)
# for p in range(parameters[6][1] - 1, -1, -1):
#     if (p + 1) not in parameters[5]:
#         distanceHospitalsMatrix = np.delete(distanceHospitalsMatrix, p, 0)
#         distanceHospitalsMatrix = np.delete(distanceHospitalsMatrix, p, 1)

distanceMatrix_ = np.array(distanceMatrix, dtype=float)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[5]:
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

clusters = assignUniformHospitals(clusters, hospitalsCoordinates, parameters[1], distanceMatrix_, distanceMatrix,
                                  redsCoordinates, len(greensCoordinates) + len(redsCoordinates))
print(clusters)

# assignment = assignUniformHospitals(len(parameters[5]), parameters[1])
#
# hospitals = {}
# for (hospital, cluster) in assignment:
#     if cluster not in hospitals:
#         hospitals[cluster] = [parameters[5][hospital]]
#     else:
#         hospitals[cluster].append(parameters[5][hospital])
#
# partitions = associateHospitals(hospitals, clusters, distanceMatrix)
#
# for (i, j) in partitions:
#     for n in hospitals[i]:
#         clusters[i][n] = coordinates[n]

# HospitalsInTeam = {}
# for (i, j) in partitions:
#     for n in hospitals[i]:
#         if i not in HospitalsInTeam:
#             HospitalsInTeam[i] = 1
#         else:
#             HospitalsInTeam[i] += 1
#
#         clusters[i][n] = coordinates[n]

# partitions, hospitalsInTeam = associateHospitals(clusters, redsInTeam, hospitalsCoordinates,
#                                                  distanceHospitalsMatrix, distanceMatrix, int(sys.argv[3]))
#
# for (i, j) in partitions:
#     for h in hospitalsInTeam[j]:
#         clusters[i][h] = coordinates[h]

finalScore = 0
finalTimeUntilLastRed = 0
scores = {}
c = 0
for i in range(0, parameters[0]):
    if (i + 1) in parameters[3]:
        scores[(i + 1)] = parameters[9][c]
        c += 1
    else:
        scores[(i + 1)] = 0.0

for cluster in clusters:
    tmp = np.array(distanceMatrix, dtype=float)
    totalNodes = []
    # mappingNodes = {}
    # nodesDeleted = 0

    # i = 0
    for p in range(parameters[6][1] - 2, 0, -1):
        if (p + 1) not in cluster:
            tmp = np.delete(tmp, p, 0)
            tmp = np.delete(tmp, p, 1)
            # nodesDeleted += 1
        else:
            # mappingNodes[i] = (p + 1)
            totalNodes.append((p + 1))
            # i += 1

    print(totalNodes)
    totalNodes.reverse()
    totalNodes = [parameters[6][0]] + totalNodes + [parameters[6][1]]

    # totalNodes = [parameters[6][0]] + totalNodes
    # totalNodes = totalNodes + [parameters[6][1]]

    # distanceMatrixforCPLEX = []
    # for i in range(0, tmp.shape[0]):
    #     for j in range(0, tmp.shape[1]):
    #         if i != j:
    #             distanceMatrixforCPLEX.append(distanceMatrix[i][j])

    hospitals = []
    reds = []
    greens = []
    numberOfReds = 0

    for n in totalNodes:
        if n in parameters[3]:
            greens.append(1.0)
        else:
            greens.append(0.0)

    for n in totalNodes:
        if n in parameters[4]:
            reds.append(1.0)
            numberOfReds += 1
        else:
            reds.append(0.0)

    for n in totalNodes:
        if n in parameters[5]:
            hospitals.append(1.0)
        else:
            hospitals.append(0.0)

    currentScores = []
    for n in totalNodes:
        currentScores.append(scores[n])

    currentServiceTime = []
    for n in totalNodes:
        currentServiceTime.append(parameters[7][n - 1])

    print("totalNodes")
    print(totalNodes)
    print("tmp")
    print(tmp)
    print("currentServiceTime")
    print(currentServiceTime)
    print("hospitals")
    print(hospitals)
    print("reds")
    print(reds)
    print("greens")
    print(greens)
    print("currentScores")
    print(currentScores)
    print("len(totalNodes)")
    print(len(totalNodes))
    print("numberOfReds")
    print(numberOfReds)
    print("parameters[8]")
    print(parameters[8])

    feasibleTour = getFeasibleTour(tmp, currentServiceTime, hospitals, reds, greens, currentScores,
                                   len(totalNodes), numberOfReds, parameters[8])

    reconstructedTour = []
    nodesInSolution = []
    for (i, j) in feasibleTour[1]:
        reconstructedTour.append((totalNodes[i], totalNodes[j]))

    tours.append(reconstructedTour)

    for n in feasibleTour[2]:
        nodesInSolution.append(totalNodes[n])

    solutionNodes.append(nodesInSolution)

    totalScore = 0
    for n in feasibleTour[2]:
        totalScore += currentScores[n]
    finalScore += totalScore

    time = 0
    for (i, j) in feasibleTour[1]:
        print(tmp[i][j])
        time += tmp[i][j]

    path = {}
    for (i, j) in reconstructedTour:
        path[i] = j
    # path[parameters[6][1] - 1] = None

    timeUntilLastRed = calculateWaitingTime(path, nodesInSolution, parameters[7], redsCoordinates, numberOfReds,
                                            distanceMatrix)

    if finalTimeUntilLastRed < timeUntilLastRed:
        finalTimeUntilLastRed = timeUntilLastRed

    firstNodeToExclude = path[1]

    for n in nodesInSolution:
        if n not in path:
            for i, j in path.items():
                if j == n:
                    secondNodeToExclude = i

    for n in nodesInSolution:
        if n != firstNodeToExclude and n != secondNodeToExclude:
            # print(currentServiceTime[n])
            time += parameters[7][n - 1]

    waitingTimes.append(timeUntilLastRed)

    print(timeUntilLastRed)
    print(feasibleTour)
    print(reconstructedTour)
    print(totalScore)
    print(time)

print("FINAL SCORE: ")
print(finalScore)
print("FINAL TIME UNTIL LAST RED: ")
print(((finalTimeUntilLastRed / 800) * 30) * (60 / 50))

localSearch(clusters, tours, parameters[7], greensCoordinates, redsCoordinates, hospitalsCoordinates, waitingTimes,
            distanceMatrix)
