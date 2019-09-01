import cvxpy as cp
import numpy as np
import scipy as sp

import os, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from colorama import Fore, Style

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def extract_coordinates(points):
    
    xs, ys = [], []
    
    for pt in points:
        xs.append(pt[0])
        ys.append(pt[1])

    return xs, ys


# A robot is a vehicle used for package handoff
# It keeps track of the trails travelled by each robot
# This is useful information for the rendering
class Robot:

    def __init__(self, xinit, yinit, speed): 
        self.speed = speed 
        self.trail = [ np.asarray([xinit, yinit]) ] 

    def current_position(self):
        return self.trail[-1]

    def update_position(self, dt, pbar):
        
        pbar = np.asarray(pbar)
        phat = 1.0/np.linalg.norm(pbar) * pbar # normalize to unit just to be safe
        
        # update current position in direction of given unit vector
        currposn = self.current_position()
        newposn  = currposn + self.speed * dt * phat 
        
        # add updated position to trail
        self.trail.append(newposn)

        # return newposition incase it is needed by the caller
        return newposn 

target  = np.asarray([1.0, -2.0])
source  = np.asarray([0.0, 0.0]) 


robots      = []
dx          = 0.3
dspeed      = 2.0
alpha, beta = 0.25, 0.25
m           = 3.0

for i in range(3):
    robots.append( Robot(alpha + i*dx, beta + i*m*dx, 2.0 + i*dspeed ))

r = len(robots) 

# Variables for rendezvous points of robot with package
X, t = [], []
for i in range(r):
    X.append(cp.Variable(2)) # vector variable
    t.append(cp.Variable( )) # scalar variable

# Constraints 
constraints_S = [  X[0] == source ]

constraints_I = [] 
for i in range(r):
    constraints_I.append( 0.0 <= t[i] )
    constraints_I.append( t[i] >= cp.norm(robots[i].current_position()-X[i]) / robots[i].speed )

constraints_L = []
for i in range(r-1):
    constraints_L.append( t[i] + cp.norm(X[i+1] - X[i])/robots[i].speed <= t[i+1]  )

objective = cp.Minimize(  t[r-1]  + cp.norm( target - X[r-1]  )/robots[r-1].speed      )


# now we are ready to run the convex optimization solver! 
# We handoff the packages from robot 0 -> 1 -> 2 -> .. . When an ordering is 
# given you can construct the optimal tour using cvxpy. The advantage
# of using cvxpy is that the problem is internally transformed into 
# a standard form required for convex optimization solvers. 
prob = cp.Problem(objective,constraints_S + constraints_I + constraints_L)
prob.solve(solver=cp.SCS)

# Find the magnitude of the constraint violation
print "S-constraint violations"
for c in constraints_S:
    print Fore.GREEN, c.violation(), Style.RESET_ALL

print "\nI-constraint violations"
for c in constraints_I:
    print Fore.GREEN, c.violation(), Style.RESET_ALL

print "\nL-constraint violations"
for c in constraints_L:
    print Fore.CYAN, c.violation(), Style.RESET_ALL
    
print "---------------------------"

for stamp in t:
    print stamp.value
print Fore.RED , "\nTotal time required to reach the destination is ", objective.value

#----------------------------------------------------------------------------
# Time for some plotting
fig, ax = plt.subplots()
ax.set_aspect(1.0)
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([-2.0, 2.0])

# Draw the movements of the robots to the respective rendezvous points 
for i in range(r):
    xs, ys = extract_coordinates(  [robots[i].current_position(), X[i].value]  )
    ax.plot(xs,ys,'g--')

# Draw the curve along which the package travels. 
for i in range(r-1):
    xs, ys = extract_coordinates(  [ X[i].value, X[i+1].value ]  )
    ax.plot(xs,ys,'bo-', linewidth=3.0, mfc='k')

xs, ys = extract_coordinates( [X[r-1].value, target]  )
ax.plot(xs,ys,'bo-',linewidth=3.0, mfc='k')



# Draw the source, target, and initial positions of the robots as bold dots
xs,ys = extract_coordinates([source, target])
ax.plot(xs,ys, 'o', markersize=20, mec='k', mfc='#F1AB30' )
ax.text(source[0], source[1], 's', fontsize=22, horizontalalignment='center',verticalalignment='center')
ax.text(target[0], target[1], 't', fontsize=22, horizontalalignment='center',verticalalignment='center')

xs, ys = extract_coordinates( [ r.current_position() for r in robots ]  )
ax.plot(xs,ys, 'o', markersize=20, mec='k', mfc='#79EB15' )

for r in robots:
    ax.text( r.current_position()[0], r.current_position()[1], str(r.speed), 
             fontsize=10, horizontalalignment='center', verticalalignment='center' )

plt.grid(color='0.5', linestyle='--', linewidth=0.5)
plt.show()


#---------------------------------------------------------------------------

# def time_of_traversal(p,q,u):
#     assert u>0.0, "The speed passed must be greater than 0.0" 

#     p = np.asarray(p)
#     q = np.asarray(q)

#     return np.linalg.norm(p-q)/u

# # we use concentric routing for this purpose, for both ease of programming 
# # and for simplicity in provability of algorithms
# imin  = 0
# dtmin = np.inf
# for idx in range(len(robots)):
#     dt = time_of_traversal( source, robots[idx].current_position(), robots[idx].speed)
#     if dt < dtmin:
#         imin = idx
#         dtmin = dt

# # update the positions of all the robots by the time dtmin in given direction
# for idx in range(len(robots)):
#     robots[idx].update_position(dtmin, source-robots[idx].current_position())

    

