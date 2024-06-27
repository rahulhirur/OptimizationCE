# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA
import projectedBacktrackingSearch as PBS
import copy

from projectionInBox import projectionInBox
from simpleValleyObjective import simpleValleyObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23184276
    return matrnr

def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    # INCOMPLETE CODE STARTS
    nk = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

    while np.linalg.norm(xp-P.project(xp-f.gradient(xp))) > eps:
        countIter+= 1 
        xj = copy.deepcopy(xp)
        rj = f.gradient(xp)

        dj = -copy.deepcopy(rj)
        curvefail= 0
        inloopIter= 1 
        while np.linalg.norm(rj) > nk:
            da = PHA.projectedHessApprox(f, P, xp, dj)

            rho= dj.T@da

            if rho <= eps* np.linalg.norm(dj)**2 :
                curvefail = 1
                break

            tj = (np.linalg.norm(rj)**2)/ rho
            xj = xj + tj*dj
            
            r_old = copy.deepcopy(rj)
            rj = r_old + tj*da

            betaj= np.linalg.norm(rj)**2 / np.linalg.norm(r_old)**2

            dj = -rj + betaj*dj
            
            inloopIter+=1

        # Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA
import projectedBacktrackingSearch as PBS
import copy

from projectionInBox import projectionInBox
from simpleValleyObjective import simpleValleyObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23184276
    return matrnr

def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    # INCOMPLETE CODE STARTS
    nk = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

    while np.linalg.norm(xp-P.project(xp-f.gradient(xp))) > eps:
        countIter+= 1 
        xj = copy.deepcopy(xp)
        rj = f.gradient(xp)

        dj = -copy.deepcopy(rj)
        curvefail= 0
        inloopIter= 1 
        while np.linalg.norm(rj) > nk:
            da = PHA.projectedHessApprox(f, P, xp, dj)

            rho= dj.T@da

            if rho <= eps* np.linalg.norm(dj)**2 :
                curvefail = 1
                break

            tj = (np.linalg.norm(rj)**2)/ rho
            xj = xj + tj*dj
            
            r_old = copy.deepcopy(rj)
            rj = r_old + tj*da

            betaj= np.linalg.norm(rj)**2 / np.linalg.norm(r_old)**2

            dj = -rj + betaj*dj
            
            inloopIter+=1

        if rho <= eps* np.linalg.norm(dj)**2:
            dk = - f.gradient(xp)
        else:
            dk = xj - xp
            
        tk = PBS.projectedBacktrackingSearch(f, P, xp, dk)
        xp = P.project(xp + tk*dk)
        nk = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp


# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)

# print(xmin)
# should return xmin close to [[1],[1]]
            
    #     tk = PBS.projectedBacktrackingSearch(f, P, xp, dk)
    #     xp = P.project(xp + tk*dk)
    #     nk = min(0.5, np.sqrt(np.linalg.norm(xp-P.project(xp-f.gradient(xp))))) * np.linalg.norm(xp-P.project(xp-f.gradient(xp)))

    # # INCOMPLETE CODE ENDS
    # if verbose:
    #     gradx = f.gradient(xp)
    #     stationarity = np.linalg.norm(xp - P.project(xp - gradx))
    #     print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    # return xp


# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)

# print(xmin)
# should return xmin close to [[1],[1]]