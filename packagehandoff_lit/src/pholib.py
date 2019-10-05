
from colorama import Fore, Style
from matplotlib import rc
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import numpy as np
import argparse, inspect, itertools, logging
import os, time, sys
import pprint as pp, randomcolor 
import utils_algo, utils_graphics

# Algorithms
def algo_pho_exact_given_order_of_drones ( drone_info, source, target ):
    import cvxpy as cp

    source = np.asarray(source)
    target = np.asarray(target)

    r = len(drone_info) 
    source = np.asarray(source)
    target = np.asarray(target)
    
    # Variables for rendezvous points of drone with package
    X, t = [], []
    for i in range(r):
       X.append(cp.Variable(2)) # vector variable
       t.append(cp.Variable( )) # scalar variable

    # Constraints 
    constraints_S = [  X[0] == source ]

    constraints_I = [] 
    for i in range(r):
      constraints_I.append(0.0 <= t[i])
      constraints_I.append(t[i] >= cp.norm(np.asarray(drone_info[i][0])-X[i])/drone_info[i][1])

    constraints_L = []
    for i in range(r-1):
      constraints_L.append(t[i] + cp.norm(X[i+1] - X[i])/drone_info[i][1] <= t[i+1])

    objective = cp.Minimize(t[r-1]+cp.norm(target-X[r-1])/drone_info[r-1][1])

    prob = cp.Problem(objective, constraints_S + constraints_I + constraints_L)
    print Fore.CYAN
    prob.solve(solver=cp.SCS,verbose=True)
    print Style.RESET_ALL
    
    package_trail = [ np.asarray(X[i].value) for i in range(r) ] + [ target ]
    return package_trail

@<Experiments@>
# Run Handlers


class Single_PHO_Input:
    def __init__(self, drone_info = [] , source = None, target=None):
           self.drone_info = drone_info 
           self.source     = source
           self.target     = target

    def get_drone_pis (self):
           return [self.drone_info[idx][0] for idx in range(len(self.drone_info)) ]
           
    def get_drone_uis (self):
           return [self.drone_info[idx][1] for idx in range(len(self.drone_info)) ]
         
    def get_tour(self, algo, animate_tour_p=False, plot_tour_p=False):
           return algo( self.drone_info, 
                        self.source, 
                        self.target, 
                        animate_tour_p, 
                        plot_tour_p    )

    # Methods for \verb|ReverseHorseflyInput|
    def clearAllStates (self):
          self.drone_info = []
          self.source = None
          self.target = None

def single_pho_run_handler():
    import random
    def wrapperEnterRunPoints(fig, ax, run):
      def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert blue circle representing the initial position of a drone
                 print Fore.GREEN
                 newPoint = (event.xdata, event.ydata)
                 speed    = np.random.uniform() # float(raw_input('What speed do you want for the drone at '+str(newPoint)))
                 run.drone_info.append( (newPoint, speed) ) 
                 patchSize  = (xlim[1]-xlim[0])/40.0
                 print Style.RESET_ALL
                 
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='#b7e8cc', edgecolor='black'  ))

                 ax.text( newPoint[0], newPoint[1], "{:.2f}".format(speed), fontsize=15, 
                          horizontalalignment='center', verticalalignment='center' )

                 ax.set_title('Number of drones inserted: ' +\
                              str(len(run.drone_info)), fontdict={'fontsize':25})
                 
             elif event.button == 3:  
                 # Insert big red circles representing the source and target points
                 patchSize  = (xlim[1]-xlim[0])/50.0
                 if run.source is None:    
                      run.source = (event.xdata, event.ydata)  
                      ax.add_patch( mpl.patches.Circle( run.source, radius = patchSize, 
                                                        facecolor= '#ffd9d6', edgecolor='black', lw=1.0 ))
                      ax.text( run.source[0], run.source[1], 'S', fontsize=15, 
                               horizontalalignment='center', verticalalignment='center' )

                 elif run.target is None:
                      run.target = (event.xdata, event.ydata)  
                      ax.add_patch( mpl.patches.Circle( run.target, radius = patchSize, 
                                                       facecolor= '#ffd9d6', edgecolor='black', lw=1.0 ))
                      ax.text( run.target[0], run.target[1], 'T', fontsize=15, 
                               horizontalalignment='center', verticalalignment='center' )
                 else:
                       print Fore.RED, "Source and Target already set", Style.RESET_ALL
             # Clear polygon patches and set up last minute \verb|ax| tweaks
             clearAxPolygonPatches(ax)
             applyAxCorrection(ax)
             fig.canvas.draw()
      return _enterPoints

    # The key-stack argument is mutable! I am using this hack to my advantage.
    def wrapperkeyPressHandler(fig, ax, run): 
           def _keyPressHandler(event):
               if event.key in ['i', 'I']:  

                    # Select algorithm to execute
                    algo_str = raw_input(Fore.YELLOW                                             +\
                            "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                            " (odw)     One Dimensional Wavefront \n"                            +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    clearAxPolygonPatches(ax)
                    if   algo_str == 'odw':
                          tour = run.get_tour( algo_odw, plot_tour_p=True )
                    else:
                          print "Unknown option. No horsefly for you! ;-D "
                          sys.exit()
                    applyAxCorrection(ax)
                    fig.canvas.draw()
                    
               elif event.key in ['c', 'C']: 
                    # Clear canvas and states of all objects
                    run.clearAllStates()
                    ax.cla()
                                  
                    applyAxCorrection(ax)
                    ax.set_xticks([])
                    ax.set_yticks([])
                                     
                    fig.texts = []
                    fig.canvas.draw()
           return _keyPressHandler
    
    # Set up interactive canvas
    fig, ax =  plt.subplots()
    run = Single_PHO_Input()
        
    from matplotlib import rc
    
    # specify the custom font to use
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    ax.set_title("Enter drone positions, source and target onto canvas. \n \
(Enter speeds into the terminal, after inserting a drone at a particular position)")

    mouseClick   = wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick)
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    
    plt.show()

# Plotting

def plot_tour(fig, ax, figtitle, source, target, 
              drone_info, used_drones, package_trail,
              xlims=[0,1],
              ylims=[0,1],
              aspect_ratio=1.0,
              speedfontsize=4,
              speedmarkersize=10,
              sourcetargetmarkerfontsize=4,
              sourcetargetmarkersize=10 ):

    import matplotlib.ticker as ticker
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    plt.rc('font', family='serif')

    # Draw the package trail
    xs, ys = extract_coordinates(package_trail)
    ax.plot(xs,ys, 'ro', markersize=5 )
    for idx in range(len(xs)-1):
          plt.arrow( xs[idx], ys[idx], xs[idx+1]-xs[idx], ys[idx+1]-ys[idx], 
                    **{'length_includes_head': True, 
                       'width': 0.007 , 
                       'head_width':0.01, 
                       'fc': 'r', 
                       'ec': 'none',
                       'alpha': 0.8})


    # Draw the source, target, and initial positions of the robots as bold dots
    xs,ys = extract_coordinates([source, target])
    ax.plot(xs,ys, 'o', markersize=sourcetargetmarkersize, alpha=0.8, ms=10, mec='k', mfc='#F1AB30' )
    #ax.plot(xs,ys, 'k--', alpha=0.6 ) # light line connecting source and target

    ax.text(source[0], source[1], 'S', fontsize=sourcetargetmarkerfontsize,\
            horizontalalignment='center',verticalalignment='center')
    ax.text(target[0], target[1], 'T', fontsize=sourcetargetmarkerfontsize,\
            horizontalalignment='center',verticalalignment='center')

    xs, ys = extract_coordinates( [ drone_info[idx][0] for idx in range(len(drone_info)) ]  )
    ax.plot(xs,ys, 'o', markersize=speedmarkersize, alpha = 0.5, mec='None', mfc='#b7e8cc' )

    # Draw speed labels
    for idx in range(len(drone_info)):
         ax.text( drone_info[idx][0][0], drone_info[idx][0][1], format(drone_info[idx][1],'.3f'),
                  fontsize=speedfontsize, horizontalalignment='center', verticalalignment='center' )

    # Draw drone path from initial position to interception point
    for pt, idx in zip(package_trail, used_drones):
         initdroneposn = drone_info[idx][0]
         handoffpoint  = pt
    
         xs, ys = extract_coordinates([initdroneposn, handoffpoint])
         plt.arrow( xs[0], ys[0], xs[1]-xs[0], ys[1]-ys[0], 
                    **{'length_includes_head': True, 
                       'width': 0.005 , 
                       'head_width':0.02, 
                       'fc': 'b', 
                       'ec': 'none'})

    fig.suptitle(figtitle, fontsize=15)
    ax.set_title('\nMakespan: ' + format(makespan(drone_info, used_drones, package_trail),'.5f'), fontsize=8)

    startx, endx = ax.get_xlim()
    starty, endy = ax.get_ylim()


    ax.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticks
                bottom='off') # turn off bottom ticks
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.1', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')

    #ax.xaxis.set_ticks(np.arange(startx, endx, 0.4))
    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
     
    #ax.yaxis.set_ticks(np.arange(starty, endy, 0.4))
    #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

    #plt.yticks(fontsize=5, rotation=90)
    #plt.xticks(fontsize=5)

    # A light grid
    #plt.grid(color='0.5', linestyle='--', linewidth=0.5)

