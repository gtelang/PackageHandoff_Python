
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

def algo_odw(drone_info, source, target, plot_tour_p = False):

    from scipy.optimize import minimize
    source     = np.asarray(source)
    target     = np.asarray(target)
    sthat      = (target-source)/np.linalg.norm(target-source)
    numdrones  = len(drone_info)
    clock_time = 0.0  # time on the global clock

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
    
    package_reached_p      = False

    while not(package_reached_p):
         # Find a faster wavelet that meets up with the package wavelet along line $\vec{ST}$ at the earliest
            
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
                 
         # Check if package wavelet reaches target before meeting wavelet computed above. Update states accordingly
            
         time_to_target_without_handoff = np.linalg.norm((target-current_package_position))/ \
                                          drone_info[current_package_handler_idx][1]

         if time_to_target_without_handoff < tI_min : 
              package_reached_p = True
              package_trail_straight.append(target)

         else:
              # Update package information (current speed, position etc.) and drone information (available and used drones)
                 
              package_handler_speed    = drone_info[current_package_handler_idx][1] 
              current_package_position = current_package_position + \
                                          package_handler_speed * (tI_min - clock_time) *  sthat
              package_trail_straight.append(current_package_position)

              clock_time                  = tI_min 
              current_package_handler_idx = idx_tI_min

              drone_pool.remove(idx_tI_min)
              used_drones.append(idx_tI_min)  
              
         

    # Run the convex optimization solver to retrieve the exact tour \verb|package_trail_cvx| for given drone order
       
    package_trail_cvx =  algo_pho_exact_given_order_of_drones(\
                                 [drone_info[idx] for idx in used_drones],source,target)

    mspan_straight    = makespan(drone_info, used_drones, package_trail_straight)
    mspan_cvx         = makespan(drone_info, used_drones, package_trail_cvx)
          
    # Plot tour if \verb|plot_tour_p == True|
       
    if plot_tour_p:
         fig0, ax0 = plt.subplots()
         plot_tour(fig0, ax0, "ODW: Straight Line", source, target, \
                   drone_info, used_drones, package_trail_straight)

         fig1, ax1 = plt.subplots()
         plot_tour(fig1, ax1, "ODW: Straight Line, Post Convex Optimization", source, target, \
                   drone_info, used_drones, package_trail_cvx)
         plt.show()
    

    return used_drones, package_trail_straight, mspan_straight, package_trail_cvx, mspan_cvx
 
def get_interception_time(s, us, p, up, t, t0) :
    
    # Change coordinates to make $s=(0,0)$ and $t$ to lie along X-axis as in \autoref{fig:getinterceptiontime}
       
    t_m = t - s # the _m subscript stands for modify
    t_m = t_m / np.linalg.norm(t_m) # normalize to unit

    # For rotating a vector clockwise by theta, 
    # to get the vector t_m into alignment with (1,0)
    costh = t_m[0]/np.sqrt(t_m[0]**2 + t_m[1]**2)
    sinth = t_m[1]/np.sqrt(t_m[0]**2 + t_m[1]**2)

    rotmat = np.asarray([[costh, sinth],
                         [-sinth, costh]])

    assert np.linalg.norm((rotmat.dot(t_m) - np.asarray([1,0]))) <= 1e-6,\
           "Rotation matrix did not work properly. t_m should get rotated\
            onto [1,0] after this transformation"

    p_shift  = p - s
    p_rot    = rotmat.dot(p_shift)
    [alpha, beta] = p_rot
    

    # Solve quadratic equation as documented in main text
    qroots = np.roots([ (1.0/us**2 - 1.0/up**2), 
                        2*t0/us + 2*alpha/up**2 , 
                        t0**2 - alpha**2/up**2 - beta**2/up**2])

    # The quadratic should always have a root. 
    qroots = np.real(qroots) # in case the imaginary parts
    qroots.sort()            # of the roots are really small

    x = None
    for root in qroots:
        if root > 0.0:
           x = root
           break
    assert abs(x/us+t0 - np.sqrt((x-alpha)**2 + beta**2)/up) <= 1e-6 , \
           "Quadratic not solved perfectly"

    tI = x/us + t0
    return tI

def time_of_travel(start, stop, speed):
     start = np.asarray(start)
     stop  = np.asarray(stop)
     return np.linalg.norm(stop-start)/speed

def extract_coordinates(points):

    xs, ys = [], []
    for pt in points:
        xs.append(pt[0])
        ys.append(pt[1])
    return np.asarray(xs), np.asarray(ys)

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
  
import networkx as nx 

def algo_matchmove(drone_info, sources, targets, plot_tour_p = False):

     # Sanity checks on input for \verb|algo_matchmove|
        
     assert len(drone_info) >= len(sources),\
        "Num drones should be >= the num source-target pairs"

     assert len(sources) == len(targets),\
        "Num sources should be == Num targets"
     
     # Basic setup
      
     sources             = [np.asarray(source) for source in sources] 
     targets             = [np.asarray(target) for target in targets] 

     drone_initposns     = [ np.asarray(initposn) for (initposn, _) in drone_info ]
     drone_speeds        = [ speed                for (_,    speed) in drone_info ]

     numpackages         = len(sources) 
     numdrones           = len(drone_info)

     package_delivered_p = [ False for _ in range(numpackages) ] 
     drone_pool          = range(numdrones)
     global_clock_time   = 0.0

     #.............................................................................
     drone_wavelets_info = [ [{'wavelet_center'        : posn,
                               'clock_time'            : 0.0,
                               'matched_package_ids'   : []}] 
                             for posn in drone_initposns ]

     def get_last_wavelet_of_drone(i):
              return drone_wavelets_info[i][-1]

     #.............................................................................
     package_trail_info  = [ [{'current_position'   : source, 
                               'clock_time'         : 0.0,
                               'current_handler_id' : None }] 
                             for source in sources ]

     def get_current_position_of_package(i):
              return package_trail_info[i][-1]['current_position']
         
     def get_current_speed_of_package(i):
              current_handler_id = package_trail_info[i][-1]['current_handler_id']

              if current_handler_id is None:
                   return 0.0
              else:
                   return drone_speeds[current_handler_id]

     
    
     while not all(package_delivered_p):
          # Construct bipartite graph $G$ on drone wavelets and package wavelets
             
          # Create nodes of the graph
          G = nx.Graph()
          G.add_nodes_from(['drone_'  +str(didx) for didx in range(numdrones)])
          G.add_nodes_from(['package_'+str(pidx) for pidx in range(numpackages)])

          # Add edges between nodes
          for didx in range(numdrones):
              for pidx in range(numpackages):
                   target = targets[pidx]
                   pkg    = get_current_position_of_package(pidx)
                   upkg   = get_current_speed_of_package(pidx)
              
                   wav    = get_last_wavelet_of_drone(didx)['wavelet_center']
                   dro    = wav['wavelet_center']
                   udro   = drone_speeds[didx]

                    if upkg < 1e-7: # zero testing for upkg
                       # Add edge to G
                       pass
           
                    elif udro > upkg: # It is critical that the zero testing for upkg 
                                      # has been done for time_target_to_solo to be 
                                      # computed safely
                       time_to_target_solo = np.linalg.norm((target-pkg))/ upkg
                       tI                  = get_interception_time(pkg, upkg, dro, udro, 
                                                                   target, wav['clock_time'])
                       if tI < time_to_target_solo:
                             # Add edge to G
                             pass
                    else:
                        continue
              
          sys.exit()
          
          # Get a bottleneck matching on $G$
          
          pass
          
          # Expand drone wavelets till an event of either Type \rnum{1} or Type \rnum{2} is detected
             
          pass
          
          # Update drone pool and package states
             
          pass
          

     # Run convex optimization solver to get improved tours for drones and packages
        
     pass
     
     # Plot movement of packages and drones if \verb|plot_tour_p == True |
        
     pass
      
     #return pass pass pass pass pass 

# Run Handlers

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

class Single_PHO_Input:
    def __init__(self, drone_info = [] , source = None, target=None):
           self.drone_info = drone_info 
           self.source     = source
           self.target     = target

    def get_drone_pis (self):
           return [self.drone_info[idx][0] for idx in range(len(self.drone_info)) ]
           
    def get_drone_uis (self):
           return [self.drone_info[idx][1] for idx in range(len(self.drone_info)) ]
         
    def get_tour(self, algo, plot_tour_p=False):
           return algo( self.drone_info, 
                        self.source, 
                        self.target, 
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

    xlim = utils_graphics.xlim
    ylim = utils_graphics.ylim

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





class Multiple_PHO_Input:
    def __init__(self, drone_info = [] , sources = [], targets=[]):
           self.drone_info = drone_info 
           self.sources     = sources
           self.targets     = targets

    def get_drone_pis (self):
           return [self.drone_info[idx][0] for idx in range(len(self.drone_info)) ]
           
    def get_drone_uis (self):
           return [self.drone_info[idx][1] for idx in range(len(self.drone_info)) ]
         
    def get_tour(self, algo, plot_tour_p=False):
           return algo( self.drone_info, 
                        self.sources, 
                        self.targets, 
                        plot_tour_p    )

    # Methods for \verb|ReverseHorseflyInput|
    def clearAllStates (self):
          self.drone_info = []
          self.sources = []
          self.targets = []



def multiple_pho_run_handler():
    import random
    def wrapperEnterRunPoints(fig, ax, run):
      def _enterPoints(event):
        if event.name      == 'button_press_event'          and \
           (event.button   == 1 or event.button == 3)       and \
            event.dblclick == True and event.xdata  != None and event.ydata  != None:

             if event.button == 1:  
                 # Insert circle representing the initial position of a drone
                 print Fore.GREEN
                 newPoint = (event.xdata, event.ydata)
                 speed    = np.random.uniform() # float(raw_input('What speed do you want for the drone at '+str(newPoint)))
                 run.drone_info.append( (newPoint, speed) ) 
                 patchSize  = (xlim[1]-xlim[0])/40.0
                 print Style.RESET_ALL
                 
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='#EBEBEB', edgecolor='black'  ))

                 ax.text( newPoint[0], newPoint[1], "{:.2f}".format(speed), fontsize=10, 
                          horizontalalignment='center', verticalalignment='center' )

                 ax.set_title('Number of drones inserted: ' +\
                              str(len(run.drone_info)), fontdict={'fontsize':25})
                 
             elif event.button == 3:  
                 # distinct colors, obtained from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
                 cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                 '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                 '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
                 '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'] 

                 # Insert big colored circles representing the source and target points

                 patchSize  = (xlim[1]-xlim[0])/50.0
                 if (len(run.sources) + len(run.targets)) % 2 == 0 : 
                        run.sources.append((event.xdata, event.ydata))
                        ax.add_patch( mpl.patches.Circle( run.sources[-1], radius = patchSize, 
                                                   facecolor= cols[len(run.sources) % len(cols)], edgecolor='black', lw=1.0 ))
                        ax.text( run.sources[-1][0], run.sources[-1][1], 'S'+str(len(run.sources)), fontsize=15, 
                                 horizontalalignment='center', verticalalignment='center' )

                 else :
                      run.targets.append((event.xdata, event.ydata))
                      ax.add_patch( mpl.patches.Circle( run.targets[-1], radius = patchSize, 
                                                       facecolor= cols[len(run.sources)%len(cols)], edgecolor='black', lw=1.0 ))
                      ax.text( run.targets[-1][0], run.targets[-1][1], 'T'+str(len(run.targets)), fontsize=15, 
                               horizontalalignment='center', verticalalignment='center' )

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
                            " (matchmove)     Repeated bottleneck matching \n"                            +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    clearAxPolygonPatches(ax)
                    if   algo_str == 'matchmove':
                          tour = run.get_tour( algo_matchmove, plot_tour_p=True )
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
    run = Multiple_PHO_Input()
        
    from matplotlib import rc
    
    # specify the custom font to use
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Times New Roman'

    xlim = utils_graphics.xlim
    ylim = utils_graphics.ylim

    ax.set_xlim([xlim[0], xlim[1]])
    ax.set_ylim([ylim[0], ylim[1]])
    ax.set_aspect(1.0)
    ax.set_xticks([])
    ax.set_yticks([])
          
    ax.set_title("Enter drone positions, sources and targets onto canvas.")

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
              speedfontsize=10,
              speedmarkersize=20,
              sourcetargetmarkerfontsize=15,
              sourcetargetmarkersize=20 ):

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
                       'head_width':0.025, 
                       'fc': 'r', 
                       'ec': 'none',
                       'alpha': 0.8})


    # Draw the source, target, and initial positions of the robots as bold dots
    xs,ys = extract_coordinates([source, target])
    ax.plot(xs,ys, 'o', markersize=sourcetargetmarkersize, alpha=1.0, ms=10, mec='k', mfc='#F1AB30' )
    #ax.plot(xs,ys, 'k--', alpha=0.6 ) # light line connecting source and target

    ax.text(source[0], source[1], 'S', fontsize=sourcetargetmarkerfontsize,\
            horizontalalignment='center',verticalalignment='center')
    ax.text(target[0], target[1], 'T', fontsize=sourcetargetmarkerfontsize,\
            horizontalalignment='center',verticalalignment='center')

    xs, ys = extract_coordinates( [ drone_info[idx][0] for idx in range(len(drone_info)) ]  )
    ax.plot(xs,ys, 'o', markersize=speedmarkersize, alpha = 1.0, mec='None', mfc='#b7e8cc' )

    # Draw speed labels
    for idx in range(len(drone_info)):
         ax.text( drone_info[idx][0][0], drone_info[idx][0][1], format(drone_info[idx][1],'.2f'),
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
    ax.set_title('\nMakespan: ' + format(makespan(drone_info, used_drones, package_trail),'.5f'), fontsize=16)

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

