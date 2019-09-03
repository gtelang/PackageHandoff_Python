
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
         
    def get_tour(self, algo):
           return algo( self.drone_info, self.source, self.target, 
                        animate_tour_p = False,
                        plot_tour_p    = False)

    # Methods for \verb|ReverseHorseflyInput|
    def clearAllStates (self):
          self.drone_info = []
          self.source = None
          self.target = None


class DroneWavelet:
    def __init__(self, startposn, speed, drone_idx,  clock_time):
          self.center          = startposn
          self.speed           = speed
          self.drone_label     = drone_idx
          self.last_reset_time = clock_time

    def get_center(self):
          return self.center

    def get_speed(self):
          return self.speed

    def get_associated_drone(self):
          return self.drone_label

    def wavelet_size(self, clock_time): # radius of the disk
           return (clock_time - last_reset_time) * self.speed
    
    def reset_wavelet(self, center, clock_time):
         self.center = center
         self.last_reset_time = clock_time 


# PHO Algorithms

def time_of_travel(start, stop, speed):
     start = np.asarray(start)
     stop  = np.asarray(stop)
     return np.linalg.norm(stop-start)/speed

def algo_odw(drone_info, source, target, 
             animate_tour_p = False,
             plot_tour_p    = False):

    from scipy.optimize import minimize
    print Fore.CYAN, "\n==========================================================="
    print            "Running the one dimensional wavefront algorithm (algo_odw) "
    print            "===========================================================", Style.RESET_ALL

    source = np.asarray(source)
    target = np.asarray(target)
    sthat  = (target-source)/np.linalg.norm(target-source) # unit vector pointing from source to target

    # Intialize wavelets for all drones
    numdrones  = len(drone_info)
    clock_time = 0.0

    drone_wavelets = []
    for (initposn, speed), idx in zip(drone_info, range(numdrones)):
         drone_wavelets.append(  DroneWavelet(initposn, speed, idx, clock_time)  )

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

    # Reset wavelet for the drone which reached the source the fastest
    clock_time = tmin
    drone_wavelets[imin].reset_wavelet(source, clock_time)

    current_package_handler_idx = imin
    current_package_position = source

    drone_pool = range(numdrones)
    drone_pool.remove(imin) 
    used_drones = [imin]

    package_reached_p   = False
    while not(package_reached_p):
          print "-----------------------------------------------------"
          print Fore.GREEN, "Clock Time ", clock_time , Style.RESET_ALL
          print "Drone Pool ", drone_pool, "  Used Drones ", used_drones
          print "-----------------------------------------------------"
          time_to_target_without_help =\
              np.linalg.norm((target-current_package_position))/drone_info[current_package_handler_idx][1]
          tintercept_min     = np.inf
          idx_intercept_min  = None

          for idx in drone_pool:
              # Calculate tintercept here via scipy
              u1 = drone_info[current_package_handler_idx][1]
              u2 = drone_info[idx][1]

              if u2 < u1:
                  continue # there won't be any use of the slower drone u2 in speeding up the package delivery process. 
              else: 
                p1 = np.asarray(drone_info[idx][0])
                p2 = np.asarray(drone_info[current_package_handler_idx][0]) 
                print u1, u2, "    ", p1, p2 

                d  = np.linalg.norm( p1 - p2 )
    
                veca  = target-p1
                vecb  = p2 - p1  
                costh = np.dot( veca, vecb ) / (np.linalg.norm(veca) * np.linalg.norm(vecb))

                fun    = lambda t: t[0]
                cons   = ({'type': 'ineq', 'fun': lambda t:  u2 + d - u1 * costh * (t[0]-clock_time)/t[0] },
                          {'type': 'ineq', 'fun': lambda t:  u2 - d + u1 * costh * (t[0]-clock_time)/t[0] })
    
                res = minimize(fun, [clock_time] , method='SLSQP', bounds=[(clock_time, None)], constraints=cons ) 

                if res.success:
                      print Fore.CYAN, "SLSQP Solver converged in "   , res.nit, " iterations",  Style.RESET_ALL

                else :
                      print Fore.RED, "Boo! SLSQP Solver did not converge!", Style.RESET_ALL
                      print res.status
                      print res.message 
                
                tintercept = res.x[0]
                print res.x

                if tintercept < tintercept_min:
                   tintercept_min    = tintercept
                   idx_intercept_min = idx

          if time_to_target_without_help < tintercept_min :
              package_reached_p = True
          else:

              package_handler_speed     = drone_info[current_package_handler_idx][1] 
              current_package_position  = current_package_position + \
                   package_handler_speed * (tintercept_min - clock_time) *  sthat

              clock_time = tintercept_min 

              current_package_handler_idx = idx_intercept_min
              drone_pool.remove(idx_intercept_min)
              used_drones.append(idx_intercept_min)  

              drone_wavelets[idx_intercept_min].reset_wavelet(current_package_position, clock_time)


    tour = algo_pho_exact_given_order_of_drones ( [drone_info[idx] for idx in used_drones],source,target )

    print Fore.RED, "Package has been delivered! ", " Ids of the drones used for delivery (in order of handoff) were ", used_drones, Style.RESET_ALL

    print "The tour points are "
    utils_algo.print_list([ drone_wavelets[idx].center for idx in used_drones ])

    if plot_tour_p:
         print Fore.GREEN, "Plotting the computed tour", Style.RESET_ALL
    
    if animate_tour_p:
         print Fore.CYAN, "Animating the computed tour", Style.RESET_ALL
    
    return tour

def  algo_pho_exact_given_order_of_drones ( drone_info ,source, target ):
    pass


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
                 speed    = float(random.randint(1,6))
                 run.drone_info.append( (newPoint, speed) ) 
                 patchSize  = (xlim[1]-xlim[0])/40.0
                 print Style.RESET_ALL
                 
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='#b7e8cc', edgecolor='black'  ))

                 ax.text( newPoint[0], newPoint[1], speed, fontsize=15, 
                          horizontalalignment='center', verticalalignment='center' )

                 ax.set_title('Number of drones inserted: ' + str(len(run.drone_info)), fontdict={'fontsize':25})
                 

             elif event.button == 3:  
                 # Insert big red circles representing the source and target points
    
                 patchSize  = (xlim[1]-xlim[0])/50.0
                 if run.source is None:    
                      run.source = (event.xdata, event.ydata)  
                      ax.add_patch( mpl.patches.Circle( run.source, radius = patchSize, facecolor= '#ffd9d6', edgecolor='black', lw=1.0 ))
                      ax.text( run.source[0], run.source[1], 'S', fontsize=15, horizontalalignment='center', verticalalignment='center' )

                 elif run.target is None:
                      run.target = (event.xdata, event.ydata)  
                      ax.add_patch( mpl.patches.Circle( run.target, radius = patchSize, facecolor= '#ffd9d6', edgecolor='black', lw=1.0 ))
                      ax.text( run.target[0], run.target[1], 'T', fontsize=15, horizontalalignment='center', verticalalignment='center' )
    
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
                    #algo_str = raw_input(Fore.YELLOW                                             +\
                    #        "Enter algorithm to be used to compute the tour:\n Options are:\n"   +\
                    #        " (odw)     One Dimensional Wavefront \n"                            +\
                    #        Style.RESET_ALL)

                    algo_str='odw'
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
                                  
                    utils_graphics.applyAxCorrection(ax)
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
          
    ax.set_title("Enter drone positions, source and target onto canvas. \n (Enter speeds into the terminal, after inserting a drone at a particular position)")

    mouseClick   = wrapperEnterRunPoints (fig,ax, run)
    fig.canvas.mpl_connect('button_press_event' , mouseClick)
          
    keyPress     = wrapperkeyPressHandler(fig,ax, run)
    fig.canvas.mpl_connect('key_press_event', keyPress   )
    
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


def animate_tour (sites, phi, horse_trajectories, fly_trajectories, 
                  animation_file_name_prefix, algo_name,  render_trajectory_trails_p = False):
    """ This function can handle the animation of multiple
    horses and flies even when the the fly trajectories are all squiggly
    and if the flies have to wait at the end of their trajectories. 
    
    A fly trajectory should only be a list of points! The sites are always the 
    first points on the trajectories. Any waiting for the flies, is assumed to be 
    at the end of their trajectories where it waits for the horse to come 
    and pick them up. 

    Every point on the horse trajectory stores a list of indices of the flies
    collected at the end point. (The first point just stores the dummy value None). 
    Usually these index lists will be size 1, but there may be heuristics where you 
    might want to collect a bunch of them together since they may already be waiting 
    there at the pick up point. 

    For each drone collected, a yellow circle is placed on top of it, so that 
    it is marked as collected to be able to see the progress of the visualization 
    as it goes on. 

    """
    import numpy as np
    import matplotlib.animation as animation
    from   matplotlib.patches import Circle
    import matplotlib.pyplot as plt 

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

    mspan, _ = makespan(horse_trajectories)
    ax.set_title("Algo: " + algo_name + "  Makespan: " + '%.4f' % mspan , fontsize=25)

    number_of_flies  = len(fly_trajectories)
    number_of_horses = len(horse_trajectories)
    colors           = utils_graphics.get_colors(number_of_horses, lightness=0.5)
        
    ax.set_xlabel( "Number of drones: " + str(number_of_flies) + "\n" + r"$\varphi=$ " + str(phi), fontsize=25)
    ims                = []
    
    # Constant for discretizing each segment inside the trajectories of the horses
    # and flies. 
    NUM_SUB_LEGS              = 2 # Number of subsegments within each segment of every trajectory
    
    # Arrays keeping track of the states of the horses
    horses_reached_endpt_p    = [False for i in range(number_of_horses)]
    horses_traj_num_legs      = [len(traj)-1 for traj in horse_trajectories] # the -1 is because the initial position of the horse is always included. 
    horses_current_leg_idx    = [0 for i in range(number_of_horses)]
    horses_current_subleg_idx = [0 for i in range(number_of_horses)] 
    horses_current_posn       = [traj[0]['coords'] for traj in horse_trajectories]

    # List of arrays keeping track of the flies collected by the horses at any given point in time, 
    fly_idxs_collected_so_far = [[] for i in range(number_of_horses)] 

    # Arrays keeping track of the states of the flies
    flies_reached_endpt_p    = [False for i in range(number_of_flies)]
    flies_traj_num_legs      = [len(traj)-1 for traj in fly_trajectories]
    flies_current_leg_idx    = [0 for i in range(number_of_flies)]
    flies_current_subleg_idx = [0 for i in range(number_of_flies)] 
    flies_current_posn       = [traj[0] for traj in fly_trajectories]

    # The drone collection process ends, when all the flies AND horses 
    # have reached their ends. Some heuristics, might involve the flies 
    # or the horses waiting at the endpoints of their respective trajectories. 
    image_frame_counter = 0
    while not(all(horses_reached_endpt_p + flies_reached_endpt_p)): 

        # Update the states of all the horses
        for hidx in range(number_of_horses):
            if horses_reached_endpt_p[hidx] == False:
                htraj                       = [elt['coords'] for elt in horse_trajectories[hidx]]
                all_flys_collected_by_horse = [i             for elt in horse_trajectories[hidx] for i in elt['fly_idxs_picked_up']]

                if horses_current_subleg_idx[hidx] <= NUM_SUB_LEGS-2:

                    horses_current_subleg_idx[hidx] += 1     # subleg idx changes
                    legidx    = horses_current_leg_idx[hidx] # the legidx remains the same
                    
                    sublegidx = horses_current_subleg_idx[hidx] # shorthand for easier reference in the next two lines
                    xcurr = np.linspace( htraj[legidx][0], htraj[legidx+1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( htraj[legidx][1], htraj[legidx+1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    horses_current_posn[hidx]  = [xcurr, ycurr] 

                    
                else:
                    horses_current_subleg_idx[hidx] = 0 # reset to 0
                    horses_current_leg_idx[hidx]   += 1 # you have passed onto the next leg
                    legidx    = horses_current_leg_idx[hidx]

                    xcurr, ycurr = htraj[legidx][0], htraj[legidx][1] # current position is the zeroth point on the next leg
                    horses_current_posn[hidx]  = [xcurr , ycurr] 

                    if horses_current_leg_idx[hidx] == horses_traj_num_legs[hidx]:
                        horses_reached_endpt_p[hidx] = True

                ####################......for marking in yellow during rendering
                fly_idxs_collected_so_far[hidx].extend(horse_trajectories[hidx][legidx]['fly_idxs_picked_up'])
                fly_idxs_collected_so_far[hidx] = list(set(fly_idxs_collected_so_far[hidx])) ### critical line, to remove duplicate elements # https://stackoverflow.com/a/7961390

        # Update the states of all the flies
        for fidx in range(number_of_flies):
            if flies_reached_endpt_p[fidx] == False:
                ftraj  = fly_trajectories[fidx]

                if flies_current_subleg_idx[fidx] <= NUM_SUB_LEGS-2:
                    
                    flies_current_subleg_idx[fidx] += 1
                    legidx    = flies_current_leg_idx[fidx]

                    sublegidx = flies_current_subleg_idx[fidx]
                    xcurr = np.linspace( ftraj[legidx][0], ftraj[legidx+1][0], NUM_SUB_LEGS+1 )[sublegidx]
                    ycurr = np.linspace( ftraj[legidx][1], ftraj[legidx+1][1], NUM_SUB_LEGS+1 )[sublegidx]
                    flies_current_posn[fidx]  = [xcurr, ycurr] 

                else:
                    flies_current_subleg_idx[fidx] = 0 # reset to zero
                    flies_current_leg_idx[fidx]   += 1 # you have passed onto the next leg
                    legidx    = flies_current_leg_idx[fidx]

                    xcurr, ycurr = ftraj[legidx][0], ftraj[legidx][1] # current position is the zeroth point on the next leg
                    flies_current_posn[fidx]  = [xcurr , ycurr] 

                    if flies_current_leg_idx[fidx] == flies_traj_num_legs[fidx]:
                        flies_reached_endpt_p[fidx] = True

        objs = []
        # Render all the horse trajectories uptil this point in time. 
        for hidx in range(number_of_horses):
            traj               = [elt['coords'] for elt in horse_trajectories[hidx]]
            current_horse_posn = horses_current_posn[hidx]
            
            if horses_current_leg_idx[hidx] != horses_traj_num_legs[hidx]: # the horse is still moving

                  xhs = [traj[k][0] for k in range(1+horses_current_leg_idx[hidx])] + [current_horse_posn[0]]
                  yhs = [traj[k][1] for k in range(1+horses_current_leg_idx[hidx])] + [current_horse_posn[1]]

            else: # The horse has stopped moving
                  xhs = [x for (x,y) in traj]
                  yhs = [y for (x,y) in traj]

            horseline, = ax.plot(xhs,yhs,'-',linewidth=5.0, markersize=6, alpha=0.80, color='#D13131')
            horseloc   = Circle((current_horse_posn[0], current_horse_posn[1]), 0.015, facecolor = '#D13131', edgecolor='k',  alpha=1.00)
            horsepatch = ax.add_patch(horseloc)
            objs.append(horseline)
            objs.append(horsepatch)

        # Render all fly trajectories uptil this point in time
        for fidx in range(number_of_flies):
            traj               = fly_trajectories[fidx]
            current_fly_posn   = flies_current_posn[fidx]
            
            if flies_current_leg_idx[fidx] != flies_traj_num_legs[fidx]: # the fly is still moving

                  xfs = [traj[k][0] for k in range(1+flies_current_leg_idx[fidx])] + [current_fly_posn[0]]
                  yfs = [traj[k][1] for k in range(1+flies_current_leg_idx[fidx])] + [current_fly_posn[1]]

            else: # The fly has stopped moving
                  xfs = [x for (x,y) in traj]
                  yfs = [y for (x,y) in traj]

            if render_trajectory_trails_p:
                flyline, = ax.plot(xfs,yfs,'-',linewidth=2.5, markersize=6, alpha=0.32, color='b')
                objs.append(flyline)


            # If the current fly is in the list of flies already collected by some horse, 
            # mark this fly with yellow. If it hasn't been serviced yet, mark it with blue
            service_status_col = 'b'
            for hidx in range(number_of_horses):
                #print fly_idxs_collected_so_far[hidx]
                if fidx in fly_idxs_collected_so_far[hidx]:
                    service_status_col = 'y'
                    break

            flyloc   = Circle((current_fly_posn[0], current_fly_posn[1]), 0.013, 
                              facecolor = service_status_col, edgecolor='k', alpha=1.00)
            flypatch = ax.add_patch(flyloc)
            objs.append(flypatch)
        
        print "........................"
        print "Appending to ims ", image_frame_counter
        ims.append(objs) 
        image_frame_counter += 1

    from colorama import Back 
   
    debug(Fore.BLACK + Back.WHITE + "\nStarted constructing ani object"+ Style.RESET_ALL)
    ani = animation.ArtistAnimation(fig, ims, interval=70, blit=True, repeat=True, repeat_delay=500)
    debug(Fore.BLACK + Back.WHITE + "\nFinished constructing ani object"+ Style.RESET_ALL)

    debug(Fore.MAGENTA + "\nStarted writing animation to disk"+ Style.RESET_ALL)
    #ani.save(animation_file_name_prefix+'.avi', dpi=100)
    #ani.save('reverse_horsefly.avi', dpi=300)
    debug(Fore.MAGENTA + "\nFinished writing animation to disk"+ Style.RESET_ALL)

    plt.show()

if __name__=="__main__":
     print "Running Package Handoff"
     single_pho_run_handler()

