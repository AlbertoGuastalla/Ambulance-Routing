import sys
import ast
import math
import numpy as np
from datetime import datetime
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
import random
import scipy.special


def getUniformPartitions(nodes, m, distanceMatrix):
    problem = cplex.Cplex()
    problem.parameters.timelimit.set(300.0)
    problem.objective.set_sense(problem.objective.sense.minimize)

    names = []
    upper_bounds = []
    lower_bounds = []
    types = []
    constraint_names = []
    constraints = []
    coefficients = []

    for k in range(0, m):
        for i in range(0, nodes):
            for j in range(0, nodes):
                coefficients.append(distanceMatrix[i][j])
                names.append("x" + str(i) + str(j) + str(k))
                types.append(problem.variables.type.integer)
                upper_bounds.append(1.0)
                lower_bounds.append(0.0)

    objective = coefficients

    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          types=types,
                          names=names)

    # Constraints
    constraintsNumber = 2

    variables = []
    for k in range(0, m):
        for i in range(0, nodes):
            for j in range(i + 1, nodes):
                variables.append("x" + str(i) + str(j) + str(k))

    constraints.append([variables, ([1.0] * (nodes * (nodes - 1) * m))])

    variables = []
    for k in range(0, m):
        for i in range(0, nodes):
            for j in range(i + 1, nodes):
                variables.append("x" + str(i) + str(j) + str(k))

    constraints.append([variables, ([1.0] * (nodes * (nodes - 1) * m))])

    rhs = [scipy.special.binom(int(nodes / m), 2)] + [scipy.special.binom(int(nodes / m) + 1, 2)]

    constraint_senses = ["G"] + ["L"]

    for i in range(0, constraintsNumber):
        constraint_names.append("c" + str(i))

    problem.linear_constraints.add(lin_expr=constraints, senses=constraint_senses, rhs=rhs, names=constraint_names)

    problem.write("prob1.lp")

    # Solve the problem
    problem.solve()

    values = problem.solution.get_values()

    for i in range(0, len(variables)):
        if (round(values[i]) == 1.0):
            print(variables[i] + " = " + str(values[i]))


def changeOp(currentOp):
    return (currentOp + 1) % 6


def calculateScore(nodesInSolution, scores):
    totalScore = 0
    for n in nodesInSolution:
        totalScore += scores[n]

    return totalScore


def calculateTotalTime(path, nodesInSolution, service_costs, distanceMatrix, startNode, endNode):
    for n1, n2 in path.items():
        if n1 == startNode:
            firstNodeToExclude = n2
        if n2 == endNode:
            secondNodeToExclude = n1

    time = 0
    for (i, j) in path.items():
        time += distanceMatrix[i - 1][j - 1]

    for n in nodesInSolution:
        if n != firstNodeToExclude and n != secondNodeToExclude:
            time += service_costs[n - 1]

    return time


def calculateWaitingTimes(path, nodesInSolution, service_costs, redsCoordinates, numberOfReds,
                         distanceMatrix, startNode, endNode):
    for n1, n2 in path.items():
        if n1 == startNode:
            firstNodeToExclude = n2
        if n2 == endNode:
            secondNodeToExclude = n1

    currentNode = 1
    timeUntilLastRed = 0
    max_completion_time = {}
    while numberOfReds > 0:
        if currentNode == firstNodeToExclude:
            # print(0.0)
            # print(tmp[currentNode][path[currentNode]])
            # print(currentNode)
            # print(path[currentNode])
            # print(distanceMatrix[currentNode - 1][path[currentNode] - 1])
            timeUntilLastRed += distanceMatrix[currentNode - 1][path[currentNode] - 1]
        else:
            # print(currentServiceTime[currentNode])
            # print(tmp[currentNode][path[currentNode]])
            # print(currentNode)
            # print(path[currentNode])
            # print(distanceMatrix[currentNode - 1][path[currentNode] - 1])
            timeUntilLastRed += distanceMatrix[currentNode - 1][path[currentNode] - 1] + service_costs[
                currentNode - 1]

        if currentNode in redsCoordinates.keys():
            max_completion_time[currentNode] = timeUntilLastRed
            if secondNodeToExclude != path[currentNode]:
                max_completion_time[currentNode] += service_costs[path[currentNode] - 1]

            numberOfReds -= 1

        currentNode = path[currentNode]

    return max_completion_time


def swapHospitals(tours, hospitalNodes, service_costs, distanceMatrix, tmax, waitingTimes, totalTimes,
                  hospitalsInSolution, redsInSolution, redNodes, firstHopitalsMap, startNode, endNode):
    min_ = math.inf
    swap = None
    totalHospitalInSolution = []

    for t in hospitalsInSolution:
        for h in hospitalsInSolution[t]:
            totalHospitalInSolution.append(h)

    for t in hospitalsInSolution:
        totalHospitalInSolution.remove(firstHopitalsMap[t])

    hospitalsNotInSolution = list(set(hospitalNodes.keys()) - set(totalHospitalInSolution))

    for t in hospitalsInSolution:
        hospitalsNotInSolution.remove(firstHopitalsMap[t])

    for h1 in totalHospitalInSolution:
        for h2 in hospitalsNotInSolution:
            for t in hospitalsInSolution:
                if h1 in hospitalsInSolution[t]:
                    newTime = teamTotalTime[t]

                    for n1, n2 in tours[t]:
                        if n2 == h1:
                            prev = n1
                        if n1 == h1:
                            next = n2

                    if prev != startNode and next != endNode:
                        gap = (distanceMatrix[prev - 1][h2 - 1] + distanceMatrix[h2 - 1][next - 1] + service_costs[
                            h2 - 1]) - (distanceMatrix[prev - 1][h1 - 1] + distanceMatrix[h1 - 1][next - 1] +
                                        service_costs[
                                            h1 - 1])
                    else:
                        gap = (distanceMatrix[prev - 1][h2 - 1] + distanceMatrix[h2 - 1][next - 1]) - (
                                distanceMatrix[prev - 1][h1 - 1] + distanceMatrix[h1 - 1][next - 1])

                    newTime = newTime + gap

                    if newTime <= tmax:
                        if min_ > newTime:
                            min_ = newTime
                            swap = (h2, h1, t)

    if swap is not None:
        newTour = []
        path = {}
        nodesInSolution = []

        for n1, n2 in tours[swap[2]]:
            i = n1
            j = n2

            if n2 == swap[1]:
                j = swap[0]
            if n1 == swap[1]:
                i = swap[0]

            if i not in nodesInSolution:
                nodesInSolution.append(i)
            if j not in nodesInSolution:
                nodesInSolution.append(j)

            path[i] = j
            newTour.append((i, j))

        hospitalsInSolution[swap[2]].append(swap[0])
        hospitalsInSolution[swap[2]].remove(swap[1])
        tours[swap[2]] = newTour
        totalTimes[swap[2]] = min_
        waitingTimes[swap[2]] = calculateWaitingTimes(path, nodesInSolution, service_costs, redNodes,
                                                      len(redsInSolution[swap[2]]), distanceMatrix, startNode, endNode)

    return tours, waitingTimes, totalTimes, swap


def interClusterMoveRed(tours, redNodes, waitingTimes, totalTimes, service_costs,
                        distanceMatrix, tmax, startNode, endNode, redsInSolution, hospitalsInSolution,
                        greensInSolution, redsTabuTagIn, redsTabuTagOut, firstHopitalsMap, iteration):
    subsets = list(itertools.combinations(range(0, len(tours)), 2))
    bestTour_t1 = bestTour_t2 = None
    swap = None

    minReds = int(len(redNodes) / len(tours))
    maxReds = minReds + 1

    min_ = math.inf
    for (t1, t2) in subsets:
        if len(redsInSolution[t1]) == minReds and len(redsInSolution[t2]) == maxReds:
            path_t1 = {}
            previous_t1 = {}
            path_t2 = {}
            previous_t2 = {}

            for (n1, n2) in tours[t1]:
                path_t1[n1] = n2
                previous_t1[n2] = n1

            for (n1, n2) in tours[t2]:
                path_t2[n1] = n2
                previous_t2[n2] = n1

            greensAndHospitals = hospitalsInSolution[t1] + greensInSolution[t1]

            for node in greensAndHospitals:
                for red2 in redsInSolution[t2]:
                    if redsTabuTagOut[red2] < iteration and redsTabuTagIn[t1][red2] < iteration:
                        p_t1 = path_t1.copy()
                        p_t2 = path_t2.copy()
                        h2 = p_t2[red2]

                        if h2 != firstHopitalsMap[t2]:
                            if (previous_t2[red2] in greensInSolution[t2] and p_t2[h2] != endNode) or (
                                    previous_t2[red2] in hospitalsInSolution[t2]):
                                p_t1[h2] = p_t1[node]
                                p_t1[node] = red2
                                p_t1[red2] = h2
                                p_t2[previous_t2[red2]] = p_t2[h2]
                                del p_t2[h2]
                                del p_t2[red2]

                                nodesInSolution_t1 = []
                                for n1, n2 in p_t1.items():
                                    if n1 not in nodesInSolution_t1:
                                        nodesInSolution_t1.append(n1)
                                    if n2 not in nodesInSolution_t1:
                                        nodesInSolution_t1.append(n2)

                                nodesInSolution_t2 = []
                                for n1, n2 in p_t2.items():
                                    if n1 not in nodesInSolution_t2:
                                        nodesInSolution_t2.append(n1)
                                    if n2 not in nodesInSolution_t2:
                                        nodesInSolution_t2.append(n2)

                                mewTime_t1 = calculateTotalTime(p_t1, nodesInSolution_t1, service_costs, distanceMatrix,
                                                                startNode, endNode)

                                newTime_t2 = calculateTotalTime(p_t2, nodesInSolution_t2, service_costs, distanceMatrix,
                                                                startNode, endNode)

                                if mewTime_t1 <= tmax and newTime_t2 <= tmax:
                                    newWaitingTime_t1 = calculateWaitingTimes(p_t1, nodesInSolution_t1, service_costs,
                                                                              redNodes,
                                                                              len(redsInSolution[t1]) + 1,
                                                                              distanceMatrix, startNode, endNode)

                                    newWaitingTime_t2 = calculateWaitingTimes(p_t2, nodesInSolution_t2, service_costs,
                                                                              redNodes,
                                                                              len(redsInSolution[t2]) - 1,
                                                                              distanceMatrix, startNode, endNode)

                                    if max(newWaitingTime_t1.values()) + max(newWaitingTime_t2.values()) < min_:
                                        min_ = max(newWaitingTime_t1.values()) + max(newWaitingTime_t2.values())
                                        w_t1 = newWaitingTime_t1
                                        w_t2 = newWaitingTime_t2
                                        bestTour_t1 = p_t1
                                        bestTour_t2 = p_t2
                                        newTotalTime_t1 = mewTime_t1
                                        newTotalTime_t2 = newTime_t2
                                        swap = (red2, t2, t1, h2)
                                        t1_ = t1
                                        t2_ = t2

    if bestTour_t1 is not None and bestTour_t2 is not None:
        newTour_t1 = []
        for n1, n2 in bestTour_t1.items():
            newTour_t1.append((n1, n2))

        newTour_t2 = []
        for n1, n2 in bestTour_t2.items():
            newTour_t2.append((n1, n2))

        tours[t1_] = newTour_t1
        tours[t2_] = newTour_t2
        waitingTimes[t1_] = w_t1
        waitingTimes[t2_] = w_t2
        totalTimes[t1_] = newTotalTime_t1
        totalTimes[t2_] = newTotalTime_t2
        redsInSolution[swap[2]].append(swap[0])
        redsInSolution[swap[1]].remove(swap[0])
        hospitalsInSolution[swap[2]].append(swap[3])
        hospitalsInSolution[swap[1]].remove(swap[3])

    return tours, waitingTimes, totalTimes, swap


def interClusterSwapReds(tours, redNodes, waitingTimes, totalTimes, service_costs, distanceMatrix,
                         tmax, startNode, endNode, redsInSolution, redsTabuTagIn, redsTabuTagOut, iteration):
    subsets = list(itertools.combinations(range(0, len(tours)), 2))
    bestTour_i = bestTour_j = None
    swap = None

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

        for red1 in redsInSolution[i]:
            for red2 in redsInSolution[j]:
                if redsTabuTagOut[red1] < iteration and redsTabuTagIn[j][red1] < iteration:
                    if redsTabuTagOut[red2] < iteration and redsTabuTagIn[i][red2] < iteration:
                        p_i = path_i.copy()
                        p_j = path_j.copy()

                        p_i[previous_i[red1]] = red2
                        p_j[previous_j[red2]] = red1

                        p_i[red2] = p_i[red1]
                        del p_i[red1]

                        p_j[red1] = p_j[red2]
                        del p_j[red2]

                        nodesInSolution_i = []
                        for n1, n2 in p_i.items():
                            if n1 not in nodesInSolution_i:
                                nodesInSolution_i.append(n1)
                            if n2 not in nodesInSolution_i:
                                nodesInSolution_i.append(n2)

                        nodesInSolution_j = []
                        for n1, n2 in p_j.items():
                            if n1 not in nodesInSolution_j:
                                nodesInSolution_j.append(n1)
                            if n2 not in nodesInSolution_j:
                                nodesInSolution_j.append(n2)

                        mewTime_i = calculateTotalTime(p_i, nodesInSolution_i, service_costs, distanceMatrix, startNode,
                                                       endNode)
                        newTime_j = calculateTotalTime(p_j, nodesInSolution_j, service_costs, distanceMatrix, startNode,
                                                       endNode)

                        if mewTime_i <= tmax and newTime_j <= tmax:
                            newWaitingTime_i = calculateWaitingTimes(p_i, nodesInSolution_i, service_costs, redNodes,
                                                                     len(redsInSolution[i]), distanceMatrix, startNode,
                                                                     endNode)

                            newWaitingTime_j = calculateWaitingTimes(p_j, nodesInSolution_j, service_costs, redNodes,
                                                                     len(redsInSolution[j]), distanceMatrix, startNode,
                                                                     endNode)

                            if max(newWaitingTime_i.values()) + max(newWaitingTime_j.values()) < min_:
                                min_ = max(newWaitingTime_i.values()) + max(newWaitingTime_j.values())
                                w_i = newWaitingTime_i
                                w_j = newWaitingTime_j
                                bestTour_i = p_i
                                bestTour_j = p_j
                                newTotalTime_i = mewTime_i
                                newTotalTime_j = newTime_j
                                swap = ((red1, i, j), (red2, j, i))
                                i_ = i
                                j_ = j

    if bestTour_i is not None and bestTour_j is not None:
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
        totalTimes[i_] = newTotalTime_i
        totalTimes[j_] = newTotalTime_j
        redsInSolution[swap[0][2]].append(swap[0][0])
        redsInSolution[swap[1][2]].append(swap[1][0])
        redsInSolution[swap[0][1]].remove(swap[0][0])
        redsInSolution[swap[1][1]].remove(swap[1][0])

    return tours, waitingTimes, totalTimes, swap


def intraClusterSwapReds(tour, redsInSolution, redNodes, waitingTime, totalTime, service_costs,
                         distanceMatrix, tmax, startNode, endNode):
    path = {}
    previous = {}
    bestTour = None
    newTotalTime = totalTime
    newTour = tour
    tmp = waitingTime

    for (i, j) in tour:
        path[i] = j
        previous[j] = i

    subsets = list(itertools.combinations(redsInSolution, 2))

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

        tt = calculateTotalTime(p, nodesInSolution, service_costs, distanceMatrix, startNode, endNode)

        if tt <= tmax:
            newWaitingTime = calculateWaitingTimes(p, nodesInSolution, service_costs, redNodes, len(redsInSolution),
                                                   distanceMatrix, startNode, endNode)

            if max(tmp.values()) > max(newWaitingTime.values()):
                newTotalTime = tt
                tmp = newWaitingTime
                bestTour = p

    if bestTour is not None:
        newTour = []
        for n1, n2 in bestTour.items():
            newTour.append((n1, n2))

    return newTour, tmp, newTotalTime


# def intraClusterSwapRedsGreens(tour, redNodes, greenNodes, waitingTime, totalTime, service_costs, distanceMatrix, tmax,
#                                startNode, endNode):
#     path = {}
#     previous = {}
#     bestTour = None
#     newTotalTime = totalTime
#     newTour = tour
#
#     for (i, j) in tour:
#         path[i] = j
#         previous[j] = i
#
#     reds = []
#     greens = []
#     for n in path.keys():
#         if n in redNodes.keys():
#             reds.append(n)
#         if n in greenNodes.keys():
#             greens.append(n)
#
#     for red in reds:
#         for green in greens:
#             p = path.copy()
#
#             p[green] = p[p[red]]
#             p[previous[green]] = red
#             p[previous[red]] = green
#             del p[p[red]]
#
#             nodesInSolution = []
#             for n1, n2 in tour:
#                 if n1 not in nodesInSolution:
#                     nodesInSolution.append(n1)
#                 if n2 not in nodesInSolution:
#                     nodesInSolution.append(n2)
#
#             print(red)
#             print(green)
#             print(path)
#             print(p)
#             tt = calculateTotalTime(p, nodesInSolution, service_costs, distanceMatrix, startNode, endNode)
#
#             if tt <= tmax:
#                 newWaitingTime = calculateWaitingTime(p, nodesInSolution, service_costs, redNodes, len(reds),
#                                                       distanceMatrix, startNode, endNode)
#
#                 if waitingTime > newWaitingTime:
#                     newTotalTime = tt
#                     waitingTime = newWaitingTime
#                     bestTour = p
#
#     if bestTour is not None:
#         newTour = []
#         for n1, n2 in bestTour.items():
#             newTour.append((n1, n2))
#
#     return newTour, waitingTime, newTotalTime


def deleteGreens(tours, greensInSolution, redsInSolution, hospitalNodes, redNodes,
                 distanceMatrix, service_costs, scores, teamScores, totalTimes, startNode, endNode):
    nodesDeleted = []

    for i in range(0, len(tours)):
        path = {}
        previous = {}

        for (n1, n2) in tours[i]:
            path[n1] = n2
            previous[n2] = n1

        for k in range(0, (int(len(greensInSolution[i]) / 3) + (len(greensInSolution[i]) % 3 > 0))):
            min_ = math.inf
            nodeToDelete = None

            for j in range(0, len(greensInSolution[i])):
                green = greensInSolution[i][j]

                previousNode = previous[green]
                nextNode = path[green]

                if previousNode not in hospitalNodes and nextNode not in hospitalNodes:
                    ratio = scores[green - 1] / (
                            service_costs[green - 1] + distanceMatrix[previousNode - 1][green - 1] +
                            distanceMatrix[green - 1][
                                nextNode - 1])

                    if min_ > ratio:
                        min_ = ratio
                        nodeToDelete = green
                        prev = previous[nodeToDelete]
                        next = path[nodeToDelete]

            if nodeToDelete != None:
                nodesDeleted.append((nodeToDelete, i))
                path[previous[nodeToDelete]] = path[nodeToDelete]
                previous[path[nodeToDelete]] = previous[nodeToDelete]
                del path[nodeToDelete]
                del previous[nodeToDelete]
                greensInSolution[i].remove(nodeToDelete)

                # if prev in hospitalsInSolution[i] and next in hospitalsInSolution[i]:
                #     a = distanceMatrix[previous[prev] - 1][prev - 1] + distanceMatrix[prev - 1][path[next] - 1]
                #     b = distanceMatrix[previous[prev] - 1][next - 1] + distanceMatrix[next - 1][path[next] - 1]
                #
                #     if a < b:
                #         totalTimes[i] = totalTimes[i] - (
                #                 distanceMatrix[prev - 1][next - 1] + distanceMatrix[next - 1][path[next] - 1]) + \
                #                         distanceMatrix[prev - 1][path[next] - 1]
                #
                #         path[prev] = path[next]
                #         del path[next]
                #         hospitalsInSolution[i].remove(next)
                #     else:
                #         totalTimes[i] = totalTimes[i] - (
                #                 distanceMatrix[previous[prev] - 1][prev - 1] + distanceMatrix[prev - 1][next - 1]) + \
                #                         distanceMatrix[previous[prev] - 1][next - 1]
                #
                #         path[previous[prev]] = next
                #         del path[prev]
                #         hospitalsInSolution[i].remove(prev)

                teamScores[i] -= scores[nodeToDelete]
                totalTimes[i] = totalTimes[i] + distanceMatrix[prev - 1][next - 1] - (
                        service_costs[nodeToDelete - 1] + distanceMatrix[prev - 1][nodeToDelete - 1] +
                        distanceMatrix[nodeToDelete - 1][next - 1])

        nodesInSolution = []
        for n1, n2 in path.items():
            if n1 not in nodesInSolution:
                nodesInSolution.append(n1)
            if n2 not in nodesInSolution:
                nodesInSolution.append(n2)

        newTour = []
        for n1, n2 in path.items():
            newTour.append((n1, n2))

        tours[i] = newTour
        waitingTimes[i] = calculateWaitingTimes(path, nodesInSolution, service_costs, redNodes,
                                                len(redsInSolution[i]), distanceMatrix, startNode,
                                                endNode)

    return tours, waitingTimes, totalTimes, teamScores, nodesDeleted


def shakeHospitals(tours, hospitalNodes, service_costs, distanceMatrix, tmax, waitingTimes, totalTimes,
                   hospitalsInSolution, redsInSolution, redNodes, firstHopitalsMap, startNode, endNode):
    swaps = []
    totalHospitalInSolution = []

    for t in hospitalsInSolution:
        for h in hospitalsInSolution[t]:
            totalHospitalInSolution.append(h)

    hospitalsNotInSolution = list(set(hospitalNodes.keys()) - set(totalHospitalInSolution))

    for t in range(0, len(tours)):
        hospitalsNotSwappedYet = hospitalsInSolution[t].copy()
        hospitalsNotSwappedYet.remove(firstHopitalsMap[t])

        for k in range(0, (int(len(hospitalsNotSwappedYet) / 3)) + (len(hospitalsNotSwappedYet) % 3 > 0)):
            min_ = math.inf
            swap = None
            rnd = random.randint(0, len(hospitalsNotSwappedYet) - 1 - k)
            h1 = hospitalsNotSwappedYet[rnd]
            hospitalsNotSwappedYet.pop(rnd)

            for n1, n2 in tours[t]:
                if n2 == h1:
                    prev = n1
                if n1 == h1:
                    next = n2

            for h2 in hospitalsNotInSolution:
                newTime = teamTotalTime[t]

                gap = (distanceMatrix[prev - 1][h2 - 1] + distanceMatrix[h2 - 1][next - 1] + service_costs[
                    h2 - 1]) - (distanceMatrix[prev - 1][h1 - 1] + distanceMatrix[h1 - 1][next - 1] +
                                service_costs[
                                    h1 - 1])

                newTime = newTime + gap

                if newTime <= tmax:
                    if min_ > newTime:
                        min_ = newTime
                        swap = (h2, h1, t)

            if swap is not None:
                swaps.append(swap)
                newTour = []
                path = {}
                nodesInSolution = []

                for n1, n2 in tours[swap[2]]:
                    i = n1
                    j = n2

                    if n2 == swap[1]:
                        j = swap[0]
                    if n1 == swap[1]:
                        i = swap[0]

                    if i not in nodesInSolution:
                        nodesInSolution.append(i)
                    if j not in nodesInSolution:
                        nodesInSolution.append(j)

                    path[i] = j
                    newTour.append((i, j))

                hospitalsInSolution[swap[2]].append(swap[0])
                hospitalsInSolution[swap[2]].remove(swap[1])
                hospitalsNotInSolution.append(swap[1])
                hospitalsNotInSolution.remove(swap[0])
                tours[swap[2]] = newTour
                totalTimes[swap[2]] = min_
                waitingTimes[swap[2]] = calculateWaitingTimes(path, nodesInSolution, service_costs, redNodes,
                                                              len(redsInSolution[swap[2]]), distanceMatrix, startNode,
                                                              endNode)

    return tours, waitingTimes, totalTimes, swaps


def insertGreen(tours, redNodes, greenNodes, redsInSolution, greensInSolution, waitingTimes, totalTimes,
                service_costs, distanceMatrix, tmax, scores, teamScores, startNode, endNode,
                greensTabuTagIn, iteration, objective):
    profit = 0
    tmp1 = math.inf
    bestTour = swap = None
    totalGreenInSolution = []

    for t in greensInSolution:
        for g in greensInSolution[t]:
            totalGreenInSolution.append(g)

    totalGreenNotInSolution = list(set(greenNodes.keys()) - set(totalGreenInSolution))

    for t in range(0, len(tours)):
        tmp = waitingTimes[t]
        path = {}

        for (i, j) in tours[t]:
            path[i] = j

        for green in totalGreenNotInSolution:
            if greensTabuTagIn[t][green] < iteration:
                for n1, n2 in path.items():
                    newTotalTime = totalTimes[t]
                    newProfit = teamScores[t] + scores[green]

                    if n1 not in redsInSolution[t] and n2 != endNode and n1 != startNode:
                        p = path.copy()
                        p[n1] = green
                        p[green] = n2

                        newTotalTime -= distanceMatrix[n1 - 1][n2 - 1]
                        newTotalTime += (distanceMatrix[n1 - 1][green - 1] + distanceMatrix[green - 1][n2 - 1])
                        newTotalTime += service_costs[green - 1]

                        if newTotalTime <= tmax:
                            nodesInSolution = []
                            for n1, n2 in p.items():
                                if n1 not in nodesInSolution:
                                    nodesInSolution.append(n1)
                                if n2 not in nodesInSolution:
                                    nodesInSolution.append(n2)

                            newWaitingTime = calculateWaitingTimes(p, nodesInSolution, service_costs, redNodes,
                                                                   len(redsInSolution[t]), distanceMatrix, startNode,
                                                                   endNode)

                            if objective == 0:
                                if max(waitingTimes[t].values()) == max(newWaitingTime.values()):
                                    flag = True
                                else:
                                    flag = False
                            else:
                                flag = True

                            if (profit < newProfit or (profit == newProfit and max(tmp.values()) > max(newWaitingTime.values())) or
                                    (profit == newProfit and max(tmp.values()) == max(newWaitingTime.values()) and tmp1 > newTotalTime)) and flag:
                                greenInserted = green
                                tmp = newWaitingTime
                                tmp1 = newTotalTime
                                profit = newProfit
                                wt = newWaitingTime
                                profitTour = newProfit
                                tourExpanded = t
                                bestTour = p
                                swap = (greenInserted, tourExpanded)

    if bestTour is not None:
        newTour = []

        for n1, n2 in bestTour.items():
            newTour.append((n1, n2))

        tours[swap[1]] = newTour
        greensInSolution[swap[1]].append(swap[0])
        waitingTimes[swap[1]] = wt
        totalTimes[swap[1]] = tmp1
        teamScores[swap[1]] = profitTour

    return tours, waitingTimes, totalTimes, teamScores, swap


def swapGreens(tours, redNodes, greenNodes, redsInSolution, greensInSolution, waitingTimes, totalTimes,
               service_costs, distanceMatrix, tmax, scores, teamScores, startNode, endNode,
               greensTabuTagIn, greensTabuTagOut, iteration, objective):
    profit = 0
    tmp1 = math.inf
    bestTour = swap = None
    totalGreenInSolution = []

    for t in greensInSolution:
        for g in greensInSolution[t]:
            totalGreenInSolution.append(g)

    totalGreenNotInSolution = list(set(greenNodes.keys()) - set(totalGreenInSolution))

    for t in range(0, len(tours)):
        tmp = waitingTimes[t]
        path = {}
        previous = {}

        for (i, j) in tours[t]:
            path[i] = j
            previous[j] = i

        for green1 in greensInSolution[t]:
            for green2 in totalGreenNotInSolution:
                if greensTabuTagIn[t][green2] < iteration and greensTabuTagOut[green1] < iteration:
                    newTotalTime = totalTimes[t] - service_costs[green1 - 1] + service_costs[green2 - 1]
                    newProfit = teamScores[t] - scores[green1] + scores[green2]

                    p = path.copy()
                    p[green2] = p[green1]
                    p[previous[green1]] = green2
                    del p[green1]

                    newTotalTime -= (distanceMatrix[previous[green1] - 1][green1 - 1] + distanceMatrix[green1 - 1][
                        path[green1] - 1])
                    newTotalTime += (
                            distanceMatrix[previous[green1] - 1][green2 - 1] + distanceMatrix[green2 - 1][
                        path[green1] - 1])

                    if newTotalTime <= tmax:
                        nodesInSolution = []
                        for n1, n2 in p.items():
                            if n1 not in nodesInSolution:
                                nodesInSolution.append(n1)
                            if n2 not in nodesInSolution:
                                nodesInSolution.append(n2)

                        newWaitingTime = calculateWaitingTimes(p, nodesInSolution, service_costs, redNodes,
                                                               len(redsInSolution[t]), distanceMatrix, startNode,
                                                               endNode)
                        if objective == 0:
                            if max(waitingTimes[t].values()) == max(newWaitingTime.values()):
                                flag = True
                            else:
                                flag = False
                        else:
                            flag = True

                        if (profit < newProfit or (profit == newProfit and max(tmp.values()) > max(newWaitingTime.values())) or
                                (profit == newProfit and max(tmp.values()) == max(newWaitingTime.values()) and tmp1 > newTotalTime)) and flag:
                            greenRemoved = green1
                            greenInserted = green2
                            tmp = newWaitingTime
                            tmp1 = newTotalTime
                            profit = newProfit
                            wt = newWaitingTime
                            profitTour = newProfit
                            tourModified = t
                            bestTour = p
                            swap = (greenInserted, greenRemoved, tourModified)

    if bestTour is not None:
        newTour = []

        for n1, n2 in bestTour.items():
            newTour.append((n1, n2))

        tours[swap[2]] = newTour
        greensInSolution[swap[2]].append(swap[0])
        greensInSolution[swap[2]].remove(swap[1])
        waitingTimes[swap[2]] = wt
        totalTimes[swap[2]] = tmp1
        teamScores[swap[2]] = profitTour

    return tours, waitingTimes, totalTimes, teamScores, swap


def localSearch(clusters, tours, service_costs, greenNodes, redNodes, hospitalNodes, nodeScores, waitingTimes,
                distanceMatrix, tmax, greensInSolution, redsInSolution, hospitalsInSolution, teamScores, teamTotalTimes,
                startNode, endNode, firstHopitalsMap, objective):
    bestTours = []
    bestRedsInSolution = {}
    bestTotalTimes = teamTotalTimes.copy()

    for tour in tours:
        bestTours.append(tour.copy())

    for t in redsInSolution:
        bestRedsInSolution[t] = redsInSolution[t].copy()

    print("Local Search:")
    print()

    max_waiting_times = []
    for w in waitingTimes:
        max_waiting_times.append(max(w.values()))
    finalTimeUntilLastRed = max(max_waiting_times)

    currentSol1 = finalTimeUntilLastRed
    currentSol2 = sum(teamScores.values())

    iteration = 0
    greensTabuTagIn = {}
    greensTabuTagOut = {}
    hospitalsTabuTagOut = {}
    hospitalsTabuTagIn = {}
    redsTabuTagIn = {}
    redsTabuTagOut = {}

    NUMBER_OF_NOT_IMPROVING_SOLUTIONS = 0
    CURRENT_OP = 0

    for i in range(0, len(tours)):
        greensTabuTagIn[i] = {}
        redsTabuTagIn[i] = {}
        hospitalsTabuTagIn[i] = {}

        for g in greenNodes:
            greensTabuTagIn[i][g] = -1

        for r in redNodes:
            redsTabuTagIn[i][r] = -1

        for h in hospitalNodes:
            hospitalsTabuTagIn[i][h] = -1

    for g in greenNodes:
        greensTabuTagOut[g] = -1

    for r in redNodes:
        redsTabuTagOut[r] = -1

    for h in hospitalNodes:
        hospitalsTabuTagOut[h] = -1

    while iteration < 100000:
        if CURRENT_OP == 0:
            # print("intraClusterSwapReds")
            for i in range(0, len(tours)):
                tours[i], waitingTimes[i], teamTotalTimes[i] = intraClusterSwapReds(tours[i], redsInSolution[i],
                                                                                    redNodes,
                                                                                    waitingTimes[i], teamTotalTimes[i],
                                                                                    service_costs, distanceMatrix, tmax,
                                                                                    startNode, endNode)

        if CURRENT_OP == 1:
            # print("interClusterSwapReds")
            tours, waitingTimes, teamTotalTimes, swap = interClusterSwapReds(tours, redNodes, waitingTimes,
                                                                             teamTotalTimes, service_costs,
                                                                             distanceMatrix, tmax, startNode, endNode,
                                                                             redsInSolution, redsTabuTagIn,
                                                                             redsTabuTagOut, iteration)

            if swap is not None:
                # print(swap)
                redsTabuTagIn[swap[0][1]][swap[0][0]] = iteration + 10
                redsTabuTagOut[swap[0][0]] = iteration + 15
                redsTabuTagIn[swap[1][1]][swap[1][0]] = iteration + 10
                redsTabuTagOut[swap[1][0]] = iteration + 15

        if CURRENT_OP == 2:
            # print("swapHospitals")
            tours, waitingTimes, teamTotalTimes, swap = swapHospitals(tours, hospitalNodes, service_costs,
                                                                      distanceMatrix, tmax, waitingTimes,
                                                                      teamTotalTimes,
                                                                      hospitalsInSolution, redsInSolution, redNodes,
                                                                      firstHopitalsMap, startNode, endNode)

            if swap is not None:
                # print(swap)
                hospitalsTabuTagIn[swap[2]][swap[1]] = iteration + 5
                hospitalsTabuTagOut[swap[0]] = iteration + 7

        if CURRENT_OP == 3:
            # print("swapGreens")
            tours, waitingTimes, teamTotalTimes, teamScores, swap = swapGreens(tours, redNodes, greenNodes,
                                                                               redsInSolution, greensInSolution,
                                                                               waitingTimes,
                                                                               teamTotalTimes, service_costs,
                                                                               distanceMatrix, tmax, scores, teamScores,
                                                                               startNode, endNode, greensTabuTagIn,
                                                                               greensTabuTagOut, iteration, objective)

            if swap is not None:
                # print(swap)
                greensTabuTagIn[swap[2]][swap[1]] = iteration + 5
                greensTabuTagOut[swap[0]] = iteration + 7

        if CURRENT_OP == 4:
            # print("insertGreen")
            tours, waitingTimes, teamTotalTimes, teamScores, swap = insertGreen(tours, redNodes, greenNodes,
                                                                                redsInSolution, greensInSolution,
                                                                                waitingTimes,
                                                                                teamTotalTimes, service_costs,
                                                                                distanceMatrix,
                                                                                tmax, scores, teamScores, startNode,
                                                                                endNode, greensTabuTagIn,
                                                                                iteration, objective)

            if swap is not None:
                # print(swap)
                greensTabuTagOut[swap[0]] = iteration + 5

        if CURRENT_OP == 5:
            # print("interClusterMoveRed")
            tours, waitingTimes, teamTotalTimes, swap = interClusterMoveRed(tours, redNodes, waitingTimes,
                                                                            teamTotalTimes, service_costs,
                                                                            distanceMatrix, tmax, startNode, endNode,
                                                                            redsInSolution, hospitalsInSolution,
                                                                            greensInSolution,
                                                                            redsTabuTagIn, redsTabuTagOut,
                                                                            firstHopitalsMap, iteration)

            if swap is not None:
                # print(swap)
                redsTabuTagIn[swap[1]][swap[0]] = iteration + 5
                redsTabuTagOut[swap[0]] = iteration + 7

        if NUMBER_OF_NOT_IMPROVING_SOLUTIONS % 20 == 0 and NUMBER_OF_NOT_IMPROVING_SOLUTIONS > 0:
            # print("deleteGreens")
            tours, waitingTimes, teamTotalTimes, teamScores, nodesDeleted = deleteGreens(tours, greensInSolution,
                                                                                         redsInSolution,
                                                                                         hospitalsCoordinates,
                                                                                         redNodes, distanceMatrix,
                                                                                         service_costs, scores,
                                                                                         teamScores,
                                                                                         teamTotalTimes, startNode,
                                                                                         endNode)

            if len(nodesDeleted) > 0:
                # print(nodesDeleted)
                for swap in nodesDeleted:
                    greensTabuTagIn[swap[1]][swap[0]] = iteration + 5

            # print("shakeHospitals")
            tours, waitingTimes, teamTotalTimes, swaps = shakeHospitals(tours, hospitalNodes,
                                                                        service_costs,
                                                                        distanceMatrix, tmax,
                                                                        waitingTimes,
                                                                        teamTotalTimes,
                                                                        hospitalsInSolution,
                                                                        redsInSolution, redNodes, firstHopitalsMap,
                                                                        startNode, endNode)

            if len(swaps) > 0:
                # print(swaps)
                for swap in swaps:
                    hospitalsTabuTagIn[swap[2]][swap[1]] = iteration + 5
                    hospitalsTabuTagOut[swap[0]] = iteration + 7

        max_waiting_times = []
        for w in waitingTimes:
            max_waiting_times.append(max(w.values()))
        finalTimeUntilLastRed = max(max_waiting_times)

        sol1 = finalTimeUntilLastRed
        sol2 = sum(teamScores.values())

        if objective == 0:
            if sol1 < currentSol1:
                currentSol1 = sol1
                currentSol2 = sol2
                bestTotalTimes = teamTotalTimes.copy()
                bestRedsInSolution = {}
                bestTours = []

                for tour in tours:
                    bestTours.append(tour.copy())

                for t in redsInSolution:
                    bestRedsInSolution[t] = redsInSolution[t].copy()

                NUMBER_OF_NOT_IMPROVING_SOLUTIONS = 0
            elif sol1 == currentSol1 and sol2 > currentSol2:
                currentSol1 = sol1
                currentSol2 = sol2
                bestTotalTimes = teamTotalTimes.copy()
                bestRedsInSolution = {}
                bestTours = []

                for tour in tours:
                    bestTours.append(tour.copy())

                for t in redsInSolution:
                    bestRedsInSolution[t] = redsInSolution[t].copy()

                NUMBER_OF_NOT_IMPROVING_SOLUTIONS = 0
            else:
                NUMBER_OF_NOT_IMPROVING_SOLUTIONS += 1
        else:
            if sol2 > currentSol2:
                currentSol1 = sol1
                currentSol2 = sol2
                bestTotalTimes = teamTotalTimes.copy()
                bestRedsInSolution = {}
                bestTours = []

                for tour in tours:
                    bestTours.append(tour.copy())

                for t in redsInSolution:
                    bestRedsInSolution[t] = redsInSolution[t].copy()

                NUMBER_OF_NOT_IMPROVING_SOLUTIONS = 0
            elif sol2 == currentSol2 and sol1 < currentSol1:
                currentSol1 = sol1
                currentSol2 = sol2
                bestTotalTimes = teamTotalTimes.copy()
                bestRedsInSolution = {}
                bestTours = []

                for tour in tours:
                    bestTours.append(tour.copy())

                for t in redsInSolution:
                    bestRedsInSolution[t] = redsInSolution[t].copy()

                NUMBER_OF_NOT_IMPROVING_SOLUTIONS = 0
            else:
                NUMBER_OF_NOT_IMPROVING_SOLUTIONS += 1

        iteration += 1
        CURRENT_OP = changeOp(CURRENT_OP)
        # print(tours)

    print(currentSol1)
    print(currentSol2)

    a1 = tmax + 1
    a2 = 1

    for sc in nodeScores:
        a2 += nodeScores[sc]

    of1 = a1 * currentSol2 - currentSol1  # Efficency
    of2 = a2 * currentSol1 - currentSol2  # Fairness

    print()
    print("OF1 - FAIRNESS: " + str(of2))
    print("OF2 - EFFICENCY: " + str(of1))
    print()

    sum_waiting_times = 0
    for w in waitingTimes:
        sum_waiting_times += sum(w.values())

    print("FINAL SCORE: ")
    print(currentSol2)
    print("SUM WAITING TIMES: ")
    print(sum_waiting_times)
    print("FINAL TIME UNTIL LAST RED: ")
    print(currentSol1)

    print(waitingTimes)
    print(teamTotalTimes)

    for tour in tours:
        for n1, n2 in tour:
            print("Nodo: " + str(n1) + " Costo: " + str(service_costs[n1 - 1]))
            print("Costo " + str(n1) + " - " + str(n2) + " : " + str(distanceMatrix[n1 - 1][n2 - 1]))


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


def assignUniformNodes(coordinates, m, partialDistanceMatrix, completeDistanceMatrix):
    # MAX_CYCLE = 1000
    flag = True
    # mapPartitionsClusters = {}
    clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
    # cycle = 0

    # if partitionElements > 0:
    #     bestCombination = assignUniform(partitions, clusters, completeDistanceMatrix)
    #     print(bestCombination)
    #
    #     for (p, c) in bestCombination:
    #         mapPartitionsClusters[p] = c
    #         partitions[p] = {**clusters[c], **partitions[p]}
    #         print(partitions[p])
    # else:

    # for i in range(0, m):
    #     mapPartitionsClusters[i] = i
    #     partitions[i] = {**clusters[i], **partitions[i]}

    minNodes = int(len(coordinates) / m)
    maxNodes = minNodes + 1

    while flag:
        flag = False
        surplusClusters = []
        lackClusters = []

        for c in range(0, len(clusters)):
            if len(clusters[c]) >= maxNodes:
                surplusClusters.append(c)
            if len(clusters[c]) < minNodes:
                lackClusters.append(c)

            if len(clusters[c]) < minNodes:
                flag = True

        if flag:
            if len(lackClusters) == 0 and len(surplusClusters) > 0:
                lackClusters = list(set(range(0, len(clusters))) - set(surplusClusters))

            if len(surplusClusters) > 0:
                min_ = math.inf
                # cycle += 1

                for c in surplusClusters:
                    for c1 in lackClusters:
                        for n in clusters[c]:
                            distance = 0
                            for n1 in clusters[c1]:
                                distance += completeDistanceMatrix[n - 1][n1 - 1]

                            distance /= (len(clusters[c1]) + 1)

                            if min_ > distance:
                                min_ = distance
                                swap = (n, c, c1)

                clusters[swap[2]][swap[0]] = clusters[swap[1]][swap[0]]
                del clusters[swap[1]][swap[0]]

                # if cycle > MAX_CYCLE:
                #     flag = False
            else:
                flag = False

    return clusters


def assignUniformReds(partitions, coordinates, m, partialDistanceMatrix, completeDistanceMatrix):
    # MAX_CYCLE = 10000
    flag = True
    # mapPartitionsClusters = {}
    clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
    bestCombination = assignUniform(partitions, clusters, distanceMatrix)

    for (p, c) in bestCombination:
        partitions[p] = {**clusters[c], **partitions[p]}

    # cycle = 0

    # if partitionElements > 0:
    #     bestCombination = assignUniform(partitions, clusters, completeDistanceMatrix)
    #     print(bestCombination)
    #
    #     for (p, c) in bestCombination:
    #         mapPartitionsClusters[p] = c
    #         partitions[p] = {**clusters[c], **partitions[p]}
    #         print(partitions[p])
    # else:

    # for i in range(0, m):
    #     mapPartitionsClusters[i] = i
    #     partitions[i] = {**clusters[i], **partitions[i]}

    minNodes = int(len(coordinates) / m)
    maxNodes = minNodes + 1

    while flag:
        flag = False
        surplusClusters = []
        lackClusters = []

        for c in range(0, len(partitions)):
            redsInPartition = 0
            for n in partitions[c]:
                if n in coordinates:
                    redsInPartition += 1

            if redsInPartition >= maxNodes:
                surplusClusters.append(c)

            if redsInPartition < minNodes:
                lackClusters.append(c)

            if redsInPartition < minNodes:
                flag = True

        if flag:
            if len(lackClusters) == 0 and len(surplusClusters) > 0:
                lackClusters = list(set(range(0, len(partitions))) - set(surplusClusters))

            if len(surplusClusters) > 0:
                min_ = math.inf
                # cycle += 1

                for c in surplusClusters:
                    for c1 in lackClusters:
                        for n in partitions[c]:
                            if n in coordinates:
                                distance = 0
                                for n1 in partitions[c1]:
                                    distance += completeDistanceMatrix[n - 1][n1 - 1]

                                distance /= (len(partitions[c1]) + 1)

                                if min_ > distance:
                                    min_ = distance
                                    swap = (n, c, c1)

                partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
                del partitions[swap[1]][swap[0]]

            # if cycle > MAX_CYCLE:
            #     flag = False
        else:
            flag = False

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


# def assignUniformReds(coordinates, m, partialDistanceMatrix, completeDistanceMatrix):
#     flag = True
#
#     clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
#
#     minNodes = int(len(coordinates) / m)
#     maxNodes = minNodes + 1
#
#     while flag:
#         flag = False
#         surplusClusters = []
#         lackClusters = []
#
#         for c in range(0, len(clusters)):
#             if len(clusters[c]) >= maxNodes:
#                 surplusClusters.append(c)
#             if len(clusters[c]) < minNodes:
#                 lackClusters.append(c)
#
#             if len(clusters[c]) > maxNodes or len(clusters[c]) < minNodes:
#                 flag = True
#
#         if flag:
#             if len(lackClusters) == 0:
#                 lackClusters = range(0, len(clusters))
#                 lackClusters = list(set(lackClusters) - set(surplusClusters))
#
#             print(surplusClusters)
#             print(lackClusters)
#             min_ = math.inf
#
#             for c in surplusClusters:
#                 for c1 in lackClusters:
#                     for n in clusters[c]:
#                         distance = 0
#                         for n1 in clusters[c1]:
#                             distance += completeDistanceMatrix[n - 1][n1 - 1]
#
#                         distance /= len(clusters[c1])
#
#                         if min_ > distance:
#                             min_ = distance
#                             swap = (n, c, c1)
#
#             clusters[swap[2]][swap[0]] = clusters[swap[1]][swap[0]]
#             del clusters[swap[1]][swap[0]]
#
#     return clusters


def assignUniformHospitals(partitions, coordinates, m, partialDistanceMatrix, completeDistanceMatrix, redNodes):
    # MAX_CYCLE = 10000
    minHospitalsInCluster = {}
    maxHospitalsInCluster = {}
    redsInCluster = {}
    flag = True
    # cycle = 0

    clusters = clusterizeData(4, partialDistanceMatrix, coordinates, m)
    bestCombination = assignUniform(partitions, clusters, distanceMatrix)

    for (p, c) in bestCombination:
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

    minNodes = int(len(coordinates) / m)
    maxNodes = minNodes + 1

    while flag:
        flag = False
        surplusClusters = []
        lackClusters = []
        hospitalsInClusters = {}

        for c in range(0, len(partitions)):
            hospitalsInClusters[c] = 0
            for h in partitions[c].keys():
                if h in coordinates:
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
                        if n in coordinates:
                            distance = 0
                            for n1 in partitions[c1]:
                                distance += completeDistanceMatrix[n - 1][n1 - 1]

                            distance /= (len(partitions[c1]) + 1)

                            if min_ > distance:
                                min_ = distance
                                swap = (n, c, c1)

            partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
            del partitions[swap[1]][swap[0]]
        else:
            surplusClusters = []
            lackClusters = []

            for c in range(0, len(partitions)):
                hospitalsInPartition = 0
                for n in partitions[c]:
                    if n in coordinates:
                        hospitalsInPartition += 1

                if hospitalsInPartition >= maxNodes and hospitalsInClusters[c] > minHospitalsInCluster[c]:
                    surplusClusters.append(c)

                if hospitalsInPartition < minNodes:
                    lackClusters.append(c)

                if hospitalsInPartition < minNodes:
                    flag = True

            # for c in range(0, len(partitions)):
            #     if len(partitions[c]) >= maxNodes and hospitalsInClusters[c] > minHospitalsInCluster[c]:
            #         surplusClusters.append(c)
            #     if len(partitions[c]) < minNodes:
            #         lackClusters.append(c)
            #
            #     if len(partitions[c]) > maxNodes:
            #         flag = True

            if flag:
                if len(lackClusters) == 0 and len(surplusClusters) > 0:
                    lackClusters = list(set(range(0, len(partitions))) - set(surplusClusters))

                if len(surplusClusters) > 0:
                    min_ = math.inf
                    # cycle += 1

                    for c in surplusClusters:
                        for c1 in lackClusters:
                            for n in partitions[c]:
                                if n in coordinates:
                                    distance = 0
                                    for n1 in partitions[c1]:
                                        distance += completeDistanceMatrix[n - 1][n1 - 1]

                                    distance /= (len(partitions[c1]) + 1)

                                    if min_ > distance:
                                        min_ = distance
                                        swap = (n, c, c1)

                    partitions[swap[2]][swap[0]] = partitions[swap[1]][swap[0]]
                    del partitions[swap[1]][swap[0]]

                    # if cycle > MAX_CYCLE:
                    #     flag = False
                else:
                    flag = False

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


def getFeasibleTour(distanceMatrix, service_costs, h, r, g, scores, nodes, nReds, tmax, fh, obj):
    problem = cplex.Cplex()
    problem.parameters.timelimit.set(300.0)
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

    tot_g_scores = 0
    for s in scores:
        tot_g_scores += s

    hospitals = h
    reds = r
    greens = g

    redsCoefficients = []
    if obj == 0:
        for i in range(1, nodes):
            redsCoefficients.append(-(tot_g_scores + 1) * reds[i])
    else:
        for i in range(1, nodes):
            redsCoefficients.append(-reds[i])

    # minusDistanceTimes = []
    # for i in range(0, nodes):
    #     for j in range(0, nodes):
    #         if i != j:
    #             minusDistanceTimes.append(0.50 * -distanceMatrix[i][j])

    names.append("sscores")
    types.append(problem.variables.type.integer)
    upper_bounds.append(cplex.infinity)
    lower_bounds.append(0.0)

    for i in range(0, nodes):
        names.append("y" + str(i))
        types.append(problem.variables.type.integer)
        upper_bounds.append(1.0)
        lower_bounds.append(0.0)

    for i in range(1, nodes):
        names.append("u" + str(i))
        types.append(problem.variables.type.integer)
        upper_bounds.append(nodes - 1)
        lower_bounds.append(1.0)

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

    if obj == 0:
        objective = [1.0] + [0.0] * nodes + redsCoefficients + [0.0] * (nodes ** 2 - nodes) + [0.0]
    else:
        objective = [(nReds * nodes + 1)] + [0.0] * nodes + redsCoefficients + [0.0] * (nodes ** 2 - nodes) + [0.0]

    problem.variables.add(obj=objective,
                          lb=lower_bounds,
                          ub=upper_bounds,
                          types=types,
                          names=names)

    # Constraints
    constraintsNumber = ((nodes - 1) * 2) + ((nodes - 1) ** 2 - (nodes - 1)) + 11

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
    for i in range(1, nodes):
        for j in range(1, nodes):
            if i != j:
                constraints.append(
                    [["u" + str(i), "u" + str(j), "x" + str(i) + "_" + str(j)], [1.0, -1.0, nodes - 1]])

    # Non possono esistere archi del tipo: r -> g
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                if redsGreensMatrix[i][j] == 1:
                    variables.append("x" + str(i) + "_" + str(j))
                    coefficients.append(1.0)
    constraints.append([variables, coefficients])

    # Non possono esistere archi del tipo: h -> h
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            if i != j:
                if hospitals[i] * hospitals[j] == 1:
                    variables.append("x" + str(i) + "_" + str(j))
                    coefficients.append(1.0)
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
                if reds[i] * reds[j] == 1:
                    variables.append("x" + str(i) + "_" + str(j))
                    coefficients.append(1.0)
    constraints.append([variables, coefficients])

    # Non possono esistere archi del tipo: g -> h -> g
    ghgContraints = 0
    for i in range(1, nodes - 1):
        for j in range(1, nodes - 1):
            for k in range(1, nodes - 1):
                if i != j and j != k:
                    if greens[i] * hospitals[j] * greens[k] == 1:
                        variables = []
                        coefficients = []
                        variables.append("x" + str(i) + "_" + str(j))
                        coefficients.append(1.0)
                        variables.append("x" + str(j) + "_" + str(k))
                        coefficients.append(1.0)
                        constraints.append([variables, coefficients])
                        constraintsNumber += 1
                        ghgContraints += 1

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

    # # Il nodo di partenza deve essere un h
    # variables = []
    # coefficients = []
    # for i in range(1, nodes):
    #     variables.append("x0" + "_" + str(i))
    #     coefficients.append(hospitals[i])
    # constraints.append([variables, coefficients])

    # Il nodo di partenza deve esere fh
    variables = []
    coefficients = []
    variables.append("x0_" + str(fh))
    coefficients.append(1.0)
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

    # definisco sscores come somma totale degli scores dei verdi in soluzione
    variables = []
    coefficients = []
    for i in range(1, nodes - 1):
        if greens[i] == 1:
            variables.append("y" + str(i))
            coefficients.append(scores[i])
    variables.append("sscores")
    coefficients.append(-1.0)
    constraints.append([variables, coefficients])

    rhs = ([0.0] * ((nodes - 1) * 2)) + ([(nodes - 2)] * ((nodes - 1) ** 2 - (nodes - 1))) + [0.0] + [0.0] + [
        0.0] + ([1.0] * ghgContraints) + ([1.0] * 2) + ([0.0] * 2) + [nReds] + [0.0] + [tmax] + [0.0]

    constraint_senses = (["E"] * ((nodes - 1) * 2)) + (["L"] * ((nodes - 1) ** 2 - (nodes - 1))) + ["E"] + [
        "E"] + ["E"] + (["L"] * ghgContraints) + (["E"] * 2) + (["E"] * 2) + ["E"] + ["E"] + ["L"] + ["E"]

    for i in range(0, constraintsNumber):
        constraint_names.append("c" + str(i))

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
        if (round(values[i]) == 1 and "x" in variables[i]):
            print(variables[i] + " = " + str(values[i]))
            v = variables[i][1:len(variables[i])]
            firstNode = v[0: v.index('_')]
            secondNode = v[v.index('_') + 1: len(v)]
            solution.append((int(firstNode), int(secondNode)))

    y = []
    for i in range(0, len(variables)):
        if (round(values[i]) == 1 and "y" in variables[i]):
            print(variables[i] + " = " + str(values[i]))
            v = variables[i][1:len(variables[i])]
            y.append(int(v))

    return (problem.solution.get_objective_value(), solution, y)


f = open(str(sys.argv[1]), "r")
f1 = open(str(sys.argv[2]), "r")
objective = int(sys.argv[3])

parameters = []
clusters = []
distanceMatrix = []
tours = []
waitingTimes = []
greensInSolution = {}
redsInSolution = {}
hospitalsInSolution = {}
teamScores = {}
teamTotalTime = {}
totalGreedyTime = 0.0

for i in range(0, 11):
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

for i in range(0, len(parameters[7])):
    parameters[7][i] = round(float(parameters[7][i]) * 0.045)

parameters[8] = round(int(parameters[8]) * 0.045)
print(parameters[8])
parameters[9] = ast.literal_eval(parameters[9])
print(parameters[9])
parameters[10] = ast.literal_eval(parameters[10])
print(parameters[10])

for i in range(0, parameters[1]):
    clusters.append({})
    greensInSolution[i] = []
    redsInSolution[i] = []
    hospitalsInSolution[i] = []

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

for i in range(0, len(distanceMatrix)):
    for j in range(0, len(distanceMatrix)):
        distanceMatrix[i][j] = round(float(distanceMatrix[i][j]) * 0.045)

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

distanceMatrix_ = np.array(distanceMatrix, dtype=int)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[3]:
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

# getUniformPartitions(distanceMatrix_.shape[0], parameters[1], distanceMatrix_)
clusters = assignUniformNodes(greensCoordinates, parameters[1], distanceMatrix_, distanceMatrix)

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

distanceMatrix_ = np.array(distanceMatrix, dtype=int)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[4]:
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

clusters = assignUniformReds(clusters, redsCoordinates, parameters[1], distanceMatrix_, distanceMatrix)

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

firstHopitalsMap = {}
for h in range(0, len(parameters[10])):
    for t in range(0, len(parameters[10][h])):
        if parameters[10][h][t] == 1:
            hospital = parameters[5][h]
            clusters[t][hospital] = coordinates[hospital]
            firstHopitalsMap[t] = hospital

distanceMatrix_ = np.array(distanceMatrix, dtype=int)
for p in range(parameters[6][1] - 1, -1, -1):
    if (p + 1) not in parameters[5] or (p + 1) in firstHopitalsMap.values():
        distanceMatrix_ = np.delete(distanceMatrix_, p, 0)
        distanceMatrix_ = np.delete(distanceMatrix_, p, 1)
        # nodesDeleted += 1

print(firstHopitalsMap)

partialHospitalCoordinates = hospitalsCoordinates.copy()
for t in range(0, parameters[1]):
    del partialHospitalCoordinates[firstHopitalsMap[t]]

clusters = assignUniformHospitals(clusters, partialHospitalCoordinates, parameters[1], distanceMatrix_, distanceMatrix,
                                  redsCoordinates)

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
clusterIndex = 0
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

    now = datetime.now()

    feasibleTour = getFeasibleTour(tmp, currentServiceTime, hospitals, reds, greens, currentScores,
                                   len(totalNodes), numberOfReds, parameters[8],
                                   totalNodes.index(firstHopitalsMap[clusterIndex]),
                                   objective)

    end = datetime.now()
    totalGreedyTime += (end - now).total_seconds()

    reconstructedTour = []
    nodesInSolution = []
    for (i, j) in feasibleTour[1]:
        reconstructedTour.append((totalNodes[i], totalNodes[j]))

    tours.append(reconstructedTour)

    for n in feasibleTour[2]:
        nodesInSolution.append(totalNodes[n])

    for n in nodesInSolution:
        if n in greensCoordinates:
            greensInSolution[clusterIndex].append(n)
        if n in redsCoordinates:
            redsInSolution[clusterIndex].append(n)
        if n in hospitalsCoordinates:
            hospitalsInSolution[clusterIndex].append(n)

    path = {}
    for (i, j) in reconstructedTour:
        path[i] = j
    # path[parameters[6][1] - 1] = None

    totalScore = calculateScore(nodesInSolution, scores)
    time = calculateTotalTime(path, nodesInSolution, parameters[7], distanceMatrix, parameters[6][0], parameters[6][1])
    timeUntilLastRed = calculateWaitingTimes(path, nodesInSolution, parameters[7], redsCoordinates, numberOfReds,
                                             distanceMatrix, parameters[6][0], parameters[6][1])

    teamScores[clusterIndex] = totalScore
    teamTotalTime[clusterIndex] = time
    finalScore += totalScore

    if finalTimeUntilLastRed < max(timeUntilLastRed.values()):
        finalTimeUntilLastRed = max(timeUntilLastRed.values())

    waitingTimes.append(timeUntilLastRed)
    clusterIndex += 1

sum_waiting_times = 0
for w in waitingTimes:
    sum_waiting_times += sum(w.values())

print("FINAL SCORE: ")
print(finalScore)
print("SUM WAITING TIMES: ")
print(sum_waiting_times)
print("FINAL TIME UNTIL LAST RED: ")
print(finalTimeUntilLastRed)
#print(((finalTimeUntilLastRed / 800) * 30) * (60 / 50))

a1 = parameters[8] + 1
a2 = 1

for sc in parameters[9]:
    a2 += sc

of1 = a1 * finalScore - finalTimeUntilLastRed  #Efficency
of2 = a2 * finalTimeUntilLastRed - finalScore   #Fairness

print()
print("OF1 - FAIRNESS: " + str(of2))
print("OF2 - EFFICENCY: " + str(of1))
print()

now = datetime.now()

localSearch(clusters, tours, parameters[7], greensCoordinates, redsCoordinates, hospitalsCoordinates, scores,
            waitingTimes, distanceMatrix, parameters[8], greensInSolution, redsInSolution, hospitalsInSolution,
            teamScores, teamTotalTime, parameters[6][0], parameters[6][1], firstHopitalsMap, objective)

end = datetime.now()
print("End Greedy Time: ")
print(totalGreedyTime)
print("End Local Search Time: ")
print(end - now)
