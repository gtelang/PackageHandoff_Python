
# Relevant imports for Package Handoff

from colorama import Fore, Style
from matplotlib import rc
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import argparse
import inspect 
import itertools
import logging
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint as pp
import randomcolor 
import sys
import time
import utils_algo
import utils_graphics

# PHO Data Structures

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


# PHO Algorithms

def time_of_travel(start, stop, speed):
     start = np.asarray(start)
     stop  = np.asarray(stop)
     return np.linalg.norm(stop-start)/speed

def algo_odw(drone_info, source, target, 
             animate_tour_p = False,
             plot_tour_p    = False):

    from scipy.optimize import minimize

    source = np.asarray(source)
    target = np.asarray(target)
    sthat  = (target-source)/np.linalg.norm(target-source) # unit vector pointing from source to target

    numdrones  = len(drone_info)
    clock_time = 0.0

    # Find the drone which can get to the source the quickest
    tmin = np.inf
    imin = None
    for idx in range(numdrones):
         initdroneposn = drone_info[idx][0]
         dronespeed    = drone_info[idx][1]
         tmin_idx = time_of_travel(initdroneposn, source, dronespeed)

         if tmin_idx < tmin:
             tmin = tmin_idx
             imin = idx 

    clock_time = tmin

    current_package_handler_idx = imin
    current_package_position    = source

    drone_pool = range(numdrones)
    drone_pool.remove(imin) 
    used_drones = [imin]

    package_trail_straight = [current_package_position]

    package_reached_p   = False
    while not(package_reached_p):

          time_to_target_without_help =\
              np.linalg.norm((target-current_package_position))/drone_info[current_package_handler_idx][1]

          tI_min     = np.inf
          idx_tI_min = None
          for idx in drone_pool:
              
              us = drone_info[current_package_handler_idx][1]
              up = drone_info[idx][1]

              if up <= us: # slower drones are useless, so skip rest of the iteration
                  continue 
              else: 
                s = current_package_position 
                p = np.asarray(drone_info[idx][0]) 

                tI = get_interception_time(s, us, p, up, target, clock_time)

                if tI < tI_min:
                   tI_min     = tI
                   idx_tI_min = idx

          if time_to_target_without_help < tI_min :
              package_reached_p = True
              package_trail_straight.append(target)

          else:
              package_handler_speed    = drone_info[current_package_handler_idx][1] 
              current_package_position = current_package_position + package_handler_speed * (tI_min - clock_time) *  sthat
              package_trail_straight.append(current_package_position)
    
              clock_time                  = tI_min 
              current_package_handler_idx = idx_tI_min

              drone_pool.remove(idx_tI_min)
              used_drones.append(idx_tI_min)  
   
    package_trail_cvx = algo_pho_exact_given_order_of_drones ([drone_info[idx] for idx in used_drones],source,target )
    mspan_straight = makespan(drone_info, used_drones, package_trail_straight)
    mspan_cvx      = makespan(drone_info, used_drones, package_trail_cvx)
    
    if plot_tour_p:
         fig0, ax0 = plt.subplots()
         plot_tour(fig0, ax0, "ODW: Straight Line"       , source, target, drone_info, used_drones, package_trail)

         fig1, ax1 = plt.subplots()
         plot_tour(fig1, ax1, "ODE: Straight Line, Post Convex Optimization", source, target, drone_info, used_drones, package_trail_cvx)
         plt.show()

    if animate_tour_p:
        #ptraj  = zip(package_trail_cvx, used_drones + [None])
        animate_tour(drone_info, [package_trail] , used_drones,
                     animation_file_name_prefix='package_handoff_animation.mp4', 
                     algo_name='odw',  
                     render_trajectory_trails_p=True) 
    
    return used_drones, package_trail_straight, mspan_straight, package_trail_cvx, mspan_cvx


def extract_coordinates(points):

    xs, ys = [], []
    for pt in points:
        xs.append(pt[0])
        ys.append(pt[1])
    return np.asarray(xs), np.asarray(ys)


def get_interception_time(s, us, p, up, t, t0) :
    
    t_m = t - s # the _m subscript stands for modify
    t_m = t_m / np.linalg.norm(t_m) # normalize to unit

    # For rotating a vector clockwise by theta, 
    # to get the vector t_m into alignment with (1,0)
    costh = t_m[0]/np.sqrt(t_m[0]**2 + t_m[1]**2)
    sinth = t_m[1]/np.sqrt(t_m[0]**2 + t_m[1]**2)

    rotmat = np.asarray([[costh, sinth],
                         [-sinth, costh]])
    
    assert np.linalg.norm((rotmat.dot(t_m) - np.asarray([1,0]))) <= 1e-6,\
           "Rotation matrix did not work properly. t_m should get rotated onto [1,0] after this transformation"

    p_shift  = p - s
    p_rot    = rotmat.dot(p_shift)
    [alpha, beta] = p_rot

    # Solve quadratic documented in the snippets above
    qroots = np.roots([ (1.0/us**2 - 1.0/up**2), 
               2*t0/us + 2*alpha/up**2 , 
               t0**2 - alpha**2/up**2 - beta**2/up**2])

    # The quadratic should always a root. 
    qroots = np.real(qroots) # in case the imaginary parts of the roots are really small, 
    qroots.sort()

    x = None
    for root in qroots:
        if root > 0.0:
           x = root
           break
    assert abs(x/us+t0 - np.sqrt((x-alpha)**2 + beta**2)/up) <= 1e-6 , "Quadratic not solved perfectly"

    tI = x/us + t0
    return tI

def algo_pho_exact_given_order_of_drones ( drone_info, source, target ):
    import cvxpy as cp

    source = np.asarray(source)
    target = np.asarray(target)

    r = len(drone_info) 
    source = np.asarray(source)
    target = np.asarray(target)
    
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
        constraints_I.append( t[i] >= cp.norm( np.asarray(drone_info[i][0]) - X[i]) / drone_info[i][1] )

    constraints_L = []
    for i in range(r-1):
         constraints_L.append( t[i] + cp.norm(X[i+1] - X[i])/drone_info[i][1] <= t[i+1] )

    objective = cp.Minimize(  t[r-1]  + cp.norm( target - X[r-1]  )/drone_info[r-1][1]  )

    prob = cp.Problem(objective, constraints_S + constraints_I + constraints_L)
    print Fore.CYAN
    prob.solve(solver=cp.SCS,verbose=True)
    print Style.RESET_ALL
    
    package_trail = [ np.asarray(X[i].value) for i in range(r) ] + [ target ]
    return package_trail


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

def makespan(drone_info, used_drones, package_trail):

    assert len(package_trail) == len(used_drones)+1, ""

    makespan = 0.0   
    counter  = 0    
    for idx in used_drones:
         dronespeed    = drone_info[idx][1]          

         makespan += time_of_travel(package_trail[counter],\
                                    package_trail[counter+1],
                                    dronespeed) 
         counter += 1
    
    return makespan

# PHO Run Handlers

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
                 speed    = float(raw_input('What speed do you want for the drone at '+str(newPoint)))
                 run.drone_info.append( (newPoint, speed) ) 
                 patchSize  = (xlim[1]-xlim[0])/40.0
                 print Style.RESET_ALL
                 
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='#b7e8cc', edgecolor='black'  ))

                 ax.text( newPoint[0], newPoint[1], speed, fontsize=15, 
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
                          tour = run.get_tour( algo_odw )
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

# PHO Experiments

def experiment_1():
    # generate the experiments folder
    import shutil
    import os
    import random
    from datetime import datetime
    import yaml


    #random.seed(6)
    random.seed(7)

   
    # Uniform random points
    source_init  = np.asarray([-1.0,0]) 
    target_init  = np.asarray([ 1.0,0]) 
    
    exsize = 100
    exreps = 5
    dtheta = 2.0*np.pi/exreps

    dir = 'pho_experiment_1' + '_exsize_' + str(exsize)
    if os.path.exists(dir):
       shutil.rmtree(dir)

    os.makedirs(dir)

    xs     = [ random.uniform(-1,1) for _ in range(exsize) ] 
    ys     = [ random.uniform(-1,1) for _ in range(exsize)] 

    initdroneposns = map(np.asarray, zip(xs,ys))
    utils_algo.print_list(initdroneposns)
    
    speeds = [random.uniform(0.01,1.) for _ in range(exsize)]

    drone_info = zip(initdroneposns, speeds)

    for idx in range(exreps):
        
         th    = idx * dtheta
         costh = np.cos(th)
         sinth = np.sin(th)

         rotmat = np.asarray([[costh, sinth],
                              [-sinth, costh]])
   
         source = rotmat.dot(source_init)
         target = rotmat.dot(target_init)

         run = Single_PHO_Input(drone_info, source, target)

         # Execute the algorithm and unpack the data
         used_drones,           \
         package_trail_straight,\
         mspan_straight,        \
         package_trail_cvx,      \
         mspan_cvx = run.get_tour(algo_odw, 
                                  animate_tour_p = False, 
                                  plot_tour_p    = False )     

         trivial_lower_bound = np.linalg.norm(target-source)/max(speeds) 

         print "Used Drones: "   , used_drones 
         print "\n"
         print Fore.CYAN, "Package Trail Straight: " , Style.RESET_ALL
         utils_algo.print_list(package_trail_straight)
         print "\n"
         print Fore.CYAN, "Package Trail CVX: ", Style.RESET_ALL        
         utils_algo.print_list(package_trail_cvx)

         print Fore.RED, "mspan_straight: ",  mspan_straight, "\n mspan_cvx:      ", mspan_cvx, Style.RESET_ALL
         print "Dilation Factor:(CVX to Straight) ", mspan_straight/mspan_cvx
         print ""
         print "CVX Solution within: ", Fore.CYAN,  mspan_cvx/trivial_lower_bound , Style.RESET_ALL, " of optimum"

         # Write data to YAML file
         data = {'drone_info'  : drone_info,
                 'source'      : source,
                 'target'      : target,
                 'used_drones' : used_drones, 
                 'odw_straight': {'package_trail': package_trail_straight,
                                  'makespan'     : mspan_straight},

                 'odw_cvx'     : {'package_trail': package_trail_cvx,
                                  'makespan'     : mspan_cvx },
                 'trivial_lower_bound': trivial_lower_bound}

         subdir = 'data_exsize_'+str(exsize) + '_repnum_' + str(idx)
         os.makedirs(dir + '/' + subdir)
   
         filename = dir + '/' + subdir + '/' + subdir + '.yml'
         with open(filename, 'w') as outfile:
             yaml.dump(data, outfile, default_flow_style=True)

         eps = 1e-1
         fig, ax = plt.subplots()
         plot_tour(fig, ax, "Package Handoff: ODW Heuristic", 
                   source, target, drone_info, used_drones, 
                   package_trail_cvx, 
                   xlims=[-1-eps,1+eps], 
                   ylims=[-1-eps,1+eps], 
                   aspect_ratio=1.0)
         print Fore.RED, "Saving figure to disk : ", idx, Style.RESET_ALL
         plt.savefig(dir + '/' + subdir + '/' + subdir + 
                     '.png', bbox_inches="tight", dpi=200)
         plt.savefig(dir+'/plot'+str(idx).zfill(4)+'.png', bbox_inches="tight", dpi=200 )

    import subprocess

    command =  "ffmpeg -r 4 -f image2 -s 1920x1080 -i " + dir +"/plot%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p " +\
               " " + dir+ "/" + "pho_handoff_animation_different_source_target.mp4"
    subprocess.call(command.split())
          
  
def experiment_2():
    import shutil
    import os
    import random
    from datetime import datetime
    import yaml

    random.seed(7)
   
    source  = np.asarray([0.1,0]) 
    target  = np.asarray([1,0]) 

    # Basic copy
    numdrones_bas = 30

    xs0     = [0.101                                for idx in range(numdrones_bas)] 
    ys0     = [(idx+1)/float(numdrones_bas)         for idx in range(numdrones_bas)] 
    speeds0 = [1.1 - (idx+1)/float(1.1*numdrones_bas) for idx in range(numdrones_bas)] 
    speeds0.reverse()

    xs = xs0 
    ys = ys0
    speeds = speeds0 

    initdroneposns = map(np.asarray, zip(xs,ys))
    drone_info = zip(initdroneposns, speeds)

    run = Single_PHO_Input(drone_info, source, target)

    # Execute the algorithm and unpack the data
    used_drones,           \
    package_trail_straight,\
    mspan_straight,        \
    package_trail_cvx,     \
    mspan_cvx = run.get_tour(algo_odw, 
                             animate_tour_p = False, 
                             plot_tour_p    = False )     

    trivial_lower_bound = np.linalg.norm(target-source)/max(speeds) 


    print "Used Drones: "   , used_drones 
    print "\n"
    print Fore.CYAN, "Package Trail Straight: " , Style.RESET_ALL
    utils_algo.print_list(package_trail_straight)
    print "\n"
    print Fore.CYAN, "Package Trail CVX: ", Style.RESET_ALL        
    utils_algo.print_list(package_trail_cvx)

    print Fore.RED, "mspan_straight: ",  mspan_straight, "\n mspan_cvx:      ", mspan_cvx, Style.RESET_ALL
    print "Dilation Factor:(CVX to Straight) ", mspan_straight/mspan_cvx
    print ""
    print "CVX Solution within: ", Fore.CYAN,  mspan_cvx/trivial_lower_bound , Style.RESET_ALL, " of optimum"


    eps = 1e-1
    fig, ax = plt.subplots()
    plot_tour(fig, ax, "Package Handoff: Bending Behavior", 
                source, target, drone_info, used_drones, 
                package_trail_cvx, 
                xlims=[-eps,1+eps], 
                ylims=[-eps,1+eps], 
                aspect_ratio=1.0,
                speedfontsize=1,
                speedmarkersize=4,
                sourcetargetmarkersize=19,
                sourcetargetmarkerfontsize=14)
    plt.show()
 
  
def experiment_3():
    import shutil
    import os
    import random
    from datetime import datetime
    import yaml

    random.seed(7)
    eps = 1e-1

    source  = np.asarray([0.1, 0.5]) 
    target  = np.asarray([1.0, 0.5]) 

    numdrones = 100

    config = 'bendy'
    

    if config == 'menorah':
        speeds = []
        xs     = []
        ys     = []
    
        for idx in range(numdrones):
             xs.append(0.1)
             ys.append( 0.5+(-1.07)**idx * 0.001 )
             speeds.append((idx+1)/float(40*numdrones))
        figtitle="Package Handoff: Many Inflection Points"


    elif config == 'bendy':   
        # Basic copy
        numdrones_bas = 15
 
        xs0     = [0.101                                for idx in range(numdrones_bas)] 
        ys0     = [0.5 + (idx+1)/float(2*numdrones_bas)         for idx in range(numdrones_bas)] 
        speeds0 = [1.1 - (idx+1)/float(1.2*numdrones_bas) for idx in range(numdrones_bas)] 
        speeds0.reverse()

        xs1    = [0.6 + (idx+1)/float(1.5*numdrones_bas)         for idx in range(numdrones_bas)] 
        ys1    = [1.0 - idx/float(1000000.0*numdrones_bas) for idx in range(numdrones_bas)]
        speeds1 = [1.04 + (idx+1)/float(3*numdrones_bas) for idx in range(numdrones_bas)]


        xs = xs0 + xs1
        ys = ys0 + ys1
        speeds = speeds0 + speeds1 


        figtitle="Bendy"
     




    initdroneposns = map(np.asarray, zip(xs,ys))
    drone_info     = zip(initdroneposns, speeds)

    run = Single_PHO_Input(drone_info, source, target)

    # Execute the algorithm and unpack the data
    used_drones,           \
    package_trail_straight,\
    mspan_straight,        \
    package_trail_cvx,     \
    mspan_cvx = run.get_tour(algo_odw, 
                             animate_tour_p = False, 
                             plot_tour_p    = False )     

    trivial_lower_bound = np.linalg.norm(target-source)/max(speeds) 

    print "Used Drones: "   , used_drones 
    print "\n"
    print Fore.CYAN, "Package Trail Straight: " , Style.RESET_ALL
    utils_algo.print_list(package_trail_straight)
    print "\n"
    print Fore.CYAN, "Package Trail CVX: ", Style.RESET_ALL        
    utils_algo.print_list(package_trail_cvx)

    print Fore.RED, "mspan_straight: ",  mspan_straight, "\n mspan_cvx:      ", mspan_cvx, Style.RESET_ALL
    print "Dilation Factor:(CVX to Straight) ", mspan_straight/mspan_cvx
    print ""
    print "CVX Solution within: ", Fore.CYAN,  mspan_cvx/trivial_lower_bound , Style.RESET_ALL, " of optimum\n"

    fig, ax = plt.subplots()
    plot_tour(fig, ax, figtitle, 
                source, target, drone_info, used_drones, 
                package_trail_cvx, 
                xlims=[-eps,1+eps], 
                ylims=[-eps,1+eps], 
                aspect_ratio=1.0,
                speedfontsize=7,
                speedmarkersize=16,
                sourcetargetmarkersize=19,
                sourcetargetmarkerfontsize=14)
    plt.show()
 


# Set up logging information relevant to this module
logger=logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def debug(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.debug('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))


def info(msg):
    frame,filename,line_number,function_name,lines,index=inspect.getouterframes(
        inspect.currentframe())[1]
    line=lines[0]
    indentation_level=line.find(line.lstrip())
    logger.info('{i} [{m}]'.format(
        i='.'*indentation_level, m=msg))

xlim, ylim = [0,1], [0,1]

def applyAxCorrection(ax):
      ax.set_xlim([xlim[0], xlim[1]])
      ax.set_ylim([ylim[0], ylim[1]])
      ax.set_aspect(1.0)

def clearPatches(ax):
    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)

def clearAxPolygonPatches(ax):

    # Get indices cooresponding to the polygon patches
    for index , patch in zip(range(len(ax.patches)), ax.patches):
        if isinstance(patch, mpl.patches.Polygon) == True:
            patch.remove()
    ax.lines[:]=[]
    applyAxCorrection(ax)

##################################
#### Buggy don't use! 
##################################

def animate_tour(drone_info, 
                 package_trajectories, 
                 used_drones, 
                 animation_file_name_prefix, 
                 algo_name,  
                 render_trajectory_trails_p = True):
    import matplotlib.animation as animation
    from   matplotlib.patches import Circle

    # Set up configurations and parameters for all necessary graphics
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_aspect('equal')

    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.set_yticks(np.arange(0, 1, 0.1))

    # Turn on the minor TICKS, which are required for the minor GRID
    ax.minorticks_on()

    # customize the major grid
    ax.grid(which='major', linestyle='--', linewidth='0.3', color='red')

    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.3', color='black')

    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])

    number_of_drones   = len(used_drones) # A drone trajectory consits of the id of the drone, followed by the actual trajectory trail
    number_of_packages = len(package_trajectories)
    colors             = utils_graphics.get_colors(number_of_packages, lightness=0.5)
    ims                = []
    
    # Constant for discretizing each segment inside the trajectories of the packages and drones
    NUM_SUB_LEGS              = 50 # Number of subsegments within each segment of every trajectory
    
    # Arrays keeping track of the states of the packages
    packages_reached_endpt_p    = [False        for  i    in range(number_of_packages)]
    packages_traj_num_legs      = [len(traj)-1  for  traj in package_trajectories     ] # the -1 is because the initial position of the package is always included. 
    packages_current_leg_idx    = [0            for  i    in range(number_of_packages)]
    packages_current_subleg_idx = [0            for  i    in range(number_of_packages)] 
    packages_current_posn       = [traj[0]      for traj in package_trajectories    ]

    # Arrays keeping track of the states of the drones
    drones_reached_rendz_p    = [False                for  i    in range(number_of_drones) ]
    drones_current_subleg_idx = [0                    for i     in range(number_of_drones) ] 
    drones_current_posn       = [drone_info[idx][0]   for idx   in used_drones             ]

    ptraj               = package_trajectories[0]
    drones_trajectories = zip(drones_current_posn, ptraj)

    # The drone collection process ends, when all packages reach their endpoints
    image_frame_counter = 0
    while not(all(packages_reached_endpt_p)): 
    
        #-------------------------------------
        # Update the states of all the drones
        #-------------------------------------
        for dridx in range(number_of_drones):
             if drones_reached_rendz_p[dridx] == False :
                  dtraj = drones_trajectories[dridx]
                  if drones_current_subleg_idx[dridx] <= NUM_SUB_LEGS-2:

                    drones_current_subleg_idx[dridx] += 1     # subleg idx changes

                    sublegidx = drones_current_subleg_idx[dridx] # shorthand for easier reference in the next two lines
                    xcurr = np.linspace( dtraj[0][0], dtraj[1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( dtraj[0][1], dtraj[1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    drones_current_posn[dridx]  = [xcurr, ycurr]

                  else:
                    #drones_current_subleg_idx[dridx] = 0 # reset to 0

                    xcurr, ycurr = dtraj[0][0], dtraj[0][1] # current position is the zeroth point on the next leg
                    drones_current_posn[dridx]  = [xcurr , ycurr]

                    if drones_current_subleg_idx[dridx] == NUM_SUB_LEGS-1:
                        drones_reached_rendz_p[dridx] = True



        #-----------------------------------------
        # Update the states of all the packages
        #-----------------------------------------
        for pkgidx in range(number_of_packages):
            #if packages_reached_endpt_p[pkgidx] == False:
              #if drones_reached_rendz_p[0] == True:  
              #if  np.linalg.norm( np.asarray(packages_current_posn[pkgidx]) - np.asarray(drones_current_posn[0]) ) < 0.001 :  
                  ptraj = package_trajectories[pkgidx]
                  if packages_current_subleg_idx[pkgidx] <= NUM_SUB_LEGS-2:

                    packages_current_subleg_idx[pkgidx] += 1     # subleg idx changes
                    legidx    = packages_current_leg_idx[pkgidx] # the legidx remains the same

                    sublegidx = packages_current_subleg_idx[pkgidx] # shorthand for easier reference in the next two lines
                    xcurr = np.linspace( ptraj[legidx][0], ptraj[legidx+1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( ptraj[legidx][1], ptraj[legidx+1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    packages_current_posn[pkgidx]  = [xcurr, ycurr]


                  else:
                    packages_current_subleg_idx[pkgidx] = 0 # reset to 0
                    packages_current_leg_idx[pkgidx]   += 1 # you have passed onto the next leg
                    legidx    = packages_current_leg_idx[pkgidx]

                    xcurr, ycurr = ptraj[legidx][0], ptraj[legidx][1] # current position is the zeroth point on the next leg
                    packages_current_posn[pkgidx]  = [xcurr , ycurr]

                    if packages_current_leg_idx[pkgidx] == packages_traj_num_legs[pkgidx]:
                        packages_reached_endpt_p[pkgidx] = True
 
            
         
        #----------------
        # Rendering
        #----------------
        objs = []
        # Render all the package trajectories uptil this point in time.  
        for pkgidx in range(number_of_packages):
            traj                 = package_trajectories[pkgidx]
            current_package_posn = packages_current_posn[pkgidx]

            if packages_current_leg_idx[pkgidx] != packages_traj_num_legs[pkgidx]: # the package is still moving

                  xhs = [traj[k][0] for k in range(1+packages_current_leg_idx[pkgidx])] + [current_package_posn[0]]
                  yhs = [traj[k][1] for k in range(1+packages_current_leg_idx[pkgidx])] + [current_package_posn[1]]

            else: # The package has stopped moving
                  xhs = [x for (x,y) in traj]
                  yhs = [y for (x,y) in traj]

            print "-----------------------------------------------------"
            print Fore.RED, "plotting package line", Style.RESET_ALL
            print "-----------------------------------------------------"
            packageline, = ax.plot(np.asarray(xhs),np.asarray(yhs),'o-', 
                                   linewidth=5.0,markersize=10, alpha=0.8, color='#D13131')
            packageloc   = Circle( np.asarray([current_package_posn[0], current_package_posn[1]]), 0.015, 
                                    facecolor = '#D13131', 
                                    edgecolor='k',  
                                    alpha=1.00)
            packagepatch = ax.add_patch(packageloc)
            objs.append(packageline)
            objs.append(packagepatch)

        # Render all the drone trajectories   uptil this point in time
        for dridx in range(number_of_drones):
            traj                 = drones_trajectories[dridx]
            current_drone_posn = drones_current_posn[dridx]

            if drones_current_subleg_idx[dridx] != NUM_SUB_LEGS-1 : # the drone is still moving

                  xhs = [ traj[0][0] ] + [current_drone_posn[0]]
                  yhs = [ traj[0][1] ] + [current_drone_posn[1]]

            else: # The drone has stopped moving
                  xhs = [x for (x,y) in traj]
                  yhs = [y for (x,y) in traj]

            print "-----------------------------------------------------"
            print Fore.RED, "plotting drone line", Style.RESET_ALL
            print "-----------------------------------------------------"
            droneline, = ax.plot(np.asarray(xhs),np.asarray(yhs),'o-', 
                                   linewidth=5.0,markersize=10, alpha=0.8, color='g')
            droneloc   = Circle( np.asarray([current_drone_posn[0], current_drone_posn[1]]), 0.015, 
                                    facecolor = 'g', 
                                    edgecolor='g',  
                                    alpha=1.00)
            dronepatch = ax.add_patch(droneloc)
            objs.append(droneline)
            objs.append(dronepatch)

        print "........................"
        print "Appending to ims ", image_frame_counter
        ims.append(objs) 
        image_frame_counter += 1


    debug(Fore.CYAN + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    debug(Fore.CYAN + "\nFinished constructing ani object"+ Style.RESET_ALL)

    #debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=100)
    #debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)
    
    plt.show()




if __name__=="__main__":
     #single_pho_run_handler()
     #experiment_1() 
     #experiment_2() 
     experiment_3()
