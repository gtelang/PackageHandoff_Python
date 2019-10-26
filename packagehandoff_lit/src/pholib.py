
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
from utils_algo import normalize

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

                 tI, x = get_interception_time_and_x(s, us, p, up, target, clock_time)

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
 
def get_interception_time_and_x(s, us, p, up, t, t0) :
    
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
    qroots = np.real(qroots) # in case the imaginary parts are really small
    qroots.sort()            

    x = None
    for root in qroots:
        if root > 0.0:
           x = root
           break


    #print Fore.GREEN, "speed of package: ", us
    #print Fore.GREEN, "wavelet center: ", p_rot , " wavelet speed: ", up, " t0: ", t0 


    #print Fore.RED, "Input:", s, us, p, up, t, t0 , Style.RESET_ALL
    #print Fore.RED, "Qroots: ",  qroots, Style.RESET_ALL
    #print Fore.RED, x/us+t0, np.sqrt((x-alpha)**2 + beta**2)/up, Style.RESET_ALL
    if x is not None:
          assert abs(x/us+t0 - np.sqrt((x-alpha)**2 + beta**2)/up) <= 1e-6 , "Quadratic not solved perfectly"
          tI = x/us + t0
    else:
          tI = np.inf
    
    return tI, x

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

     # State variables that change during the main loop
     package_delivered_p = [ False for _ in range(numpackages) ] 
     drone_locked_p      = [ False for _ in range(numdrones)   ]
     drone_pool          = range(numdrones)
     remaining_packages  = range(numpackages)
     global_clock_time   = 0.0
     drone_wavelets_info = [ [{'wavelet_center'        : posn,
                               'clock_time'            : 0.0,
                               'matched_package_ids'   : []}] 
                             for posn in drone_initposns ]
     package_trail_info  = [ [{'current_position'   : source, 
                               'clock_time'         : 0.0,
                               'current_handler_id' : None }] 
                             for source in sources ]

     # Useful functions for extracting information from state variables above
     def get_last_wavelet_of_drone(i):
              return drone_wavelets_info[i][-1]

     def get_current_position_of_package(i):
              return package_trail_info[i][-1]['current_position']
         
     def get_current_speed_of_package(i):
              current_handler_id = package_trail_info[i][-1]['current_handler_id']

              if current_handler_id is None:
                   return 0.0
              else:
                   return drone_speeds[current_handler_id]

     def get_current_handler_of_package(i):
              return package_trail_info[i][-1]['current_handler_id']
     
     inum = 0
     while not all(package_delivered_p):
          print Fore.GREEN, "************************************************************************************************  ", \
                  "INUM: ", inum, " GLOBAL_CLOCK_TIME: ", "{0:.4f}".format(global_clock_time), Style.RESET_ALL
          inum += 1
          
          print Fore.RED, ".........................................PACKAGE TRAIL INFO........................................."
          for (trail, panum) in zip(package_trail_info, range(len(package_trail_info))):
                 print "\n Package number ===> ", panum
                 for d in trail:
                    print "clock time: "           ,"{0:.4f}".format(d['clock_time']),\
                          "  current_position: "     , d['current_position'],\
                          "  current_handler_id: "   , d['current_handler_id']

          print Fore.CYAN, "\n......................................DRONE WAVELETS INFO........................................"
          for (drwavs, drnum) in zip(drone_wavelets_info, range(len(drone_wavelets_info))):
               print "\nDrone number ===>  ", drnum
               for wav in drwavs:
                   print "clock time: "           ,"{0:.4f}".format(wav['clock_time']),\
                         "  wavelet_center: "     , wav['wavelet_center'],\
                         "  matched_package_ids: ", wav['matched_package_ids']    

 
          # put an assert which says that the drones handling the packages are all distinct and don't intersect
          # TODO!!!!!!!!!!!!!!!!!!!!


    
          print Style.RESET_ALL
          print "Remaining Packages:", remaining_packages
          print  "Drone Pool        :", drone_pool

          #-----------------------------------------------------------
          # Construct Bipartite graph $G$ on drone wavelets and package wavelets
          infty       = 1e7 # np.inf is not accepted by the scipy solver for min-weight matching
          zerotol     = 1e-7
          G_mat       = np.full((len(remaining_packages),len(drone_pool)), infty)
          lbend_edges = []

          # dictionary mapping serial numbers into 
          D = {ctr: dr for ctr, dr in zip(range(len(drone_pool))        , drone_pool        )}
          P = {ctr: pa for ctr, pa in zip(range(len(remaining_packages)), remaining_packages)}

          # the inverted version of the dictionaries above
          ID = {dr : ctr for ctr, dr in D.iteritems()}
          IP = {pa : ctr for ctr, pa in P.iteritems()}

          for didx in range(len(drone_pool)):
              for pidx in range(len(remaining_packages)): 
                  print "==========>", Fore.YELLOW ,\
                         "\npidx                                   : ", pidx,\
                         "\nP                                      : ", P,\
                         "\nget_current_handler_pf_pacakge(P[pidx]): ", get_current_handler_of_package(P[pidx]) ,\
                         "\nID                                     : ", ID, Style.RESET_ALL
                  current_handler_of_package = ID[get_current_handler_of_package(P[pidx])] if get_current_handler_of_package(P[pidx]) is not None else None
                  target                     = targets[P[pidx]]
                  pkg   , upkg               = get_current_position_of_package(P[pidx]), get_current_speed_of_package(P[pidx])
                  wav                        = get_last_wavelet_of_drone(D[didx])
                  dro   , udro               = wav['wavelet_center']                , drone_speeds[D[didx]]

                  if  ((current_handler_of_package is None)  or current_handler_of_package != didx) and not(drone_locked_p[D[didx]]):

                          if upkg < zerotol :
                                # Insert edge incident to stationary package and a drone ...                
                                G_mat[pidx, didx] = np.linalg.norm(pkg-dro)/udro +\
                                                    np.linalg.norm(target-pkg)/udro
                                lbend_edges.append({'edge_pair': (pidx,didx), 
                                                    'y'        : np.linalg.norm(pkg-dro)/udro }) 
                                                      
                          elif udro > upkg and abs(upkg-udro) > zerotol:
                                # Insert edge incident to a moving package and a faster unlocked drone
                                   
                                time_to_target_solo = np.linalg.norm(target-pkg)/upkg
                                tI, x  = get_interception_time_and_x_generalized(pkg, upkg, dro, udro, target, 
                                                                                global_clock_time, wav['clock_time'])
                                if x is not None: 
                                   assert tI is not None, "tI and x should be None or not None simultaneously"
                                   if tI < global_clock_time + time_to_target_solo:

                                     pthat             = (target-pkg)/np.linalg.norm(target-pkg)
                                     interception_pt   = pkg + x * pthat
                                     G_mat[pidx, didx] = tI + np.linalg.norm((target-interception_pt))/udro 
                                     lbend_edges.append({'edge_pair': (pidx,didx), 
                                                         'y'       : np.linalg.norm(interception_pt-dro)/udro }) 
                                

                  elif  ((current_handler_of_package is None)  or current_handler_of_package != didx) and drone_locked_p[D[didx]]:
                         pass # drone locked, so it cant help, keep the edge weight infinite

                  elif current_handler_of_package == didx and drone_locked_p[D[didx]]:
                         assert abs(udro-upkg) < zerotol , "udro should be equal to upkg"
                         G_mat[pidx, didx] = np.linalg.norm((target-pkg))/udro                               
                  else : 
                         print didx, current_handler_of_package, drone_locked_p[D[didx]]
                         assert not(current_handler_of_package == didx and not(drone_locked_p[D[didx]])),\
                            "This else branch should not be executed. This means didx is handling a package and is NOT locked"

          #print "G_mat\n ", G_mat
                         
          # Get a bottleneck matching on $G$
          from scipy.optimize import linear_sum_assignment
          pkg_ind, dro_ind = linear_sum_assignment(G_mat)
          assert len(pkg_ind) == len(dro_ind), "Lengths of the index arrays should be the same"

          # Expand drone wavelets till an event of either Type \rnum{1} or Type \rnum{2} is detected
             
          # Classify edges according to whether they are straight edges or lbend edges
           
          lbend_edges_of_matching    = [ d for d in lbend_edges if d['edge_pair'] in zip(pkg_ind, dro_ind) ] 
          straight_edges_of_matching = list(  set(zip(pkg_ind,dro_ind)).difference(\
                                                          set([d['edge_pair'] for d in lbend_edges_of_matching]))   )
          
          # Get lowest weight edge (with its weight denoted emin) in the computed matching
          imin  = 0
          ewmin = np.inf
          for pidx, didx, i in zip(pkg_ind, dro_ind, range(len(pkg_ind))):
              edgewt = G_mat[pidx, didx]
              if edgewt < ewmin :
                    imin, ewmin  = i, edgewt
                    
          pmin, dmin = pkg_ind[imin], dro_ind[imin]
          
          # Check if there is an lbend edge in the matching which has a $y\leq ewmin$. If so, 
          # find the one with the one with the lowest such $y$
          ymin         = np.inf
          plmin, dlmin = None, None

          for ledge in lbend_edges_of_matching:
              (pl,dl) = ledge['edge_pair']
              y       = ledge['y']
              
              if y < ymin:
                  ymin, plmin, dlmin = y, pl, dl
 
          #-----------------------------------------------------------------------------------------------------------------        
          # Find the least time it takes for a non-matched drone (if one exists at this point) to reach some package.
          # such a non-matched package can only aid in the delivery time.
          non_matched_drones = list(set(drone_pool).difference(set([D[d] for d in dro_ind])))
          nonm_tmin          = np.inf
          nonm_dmin          = None
          nonm_pmin          = None
          if non_matched_drones:
               for d in non_matched_drones:
                  if drone_locked_p[d] == False: # only look at unlocked non-matched drones 
                    wav     = get_last_wavelet_of_drone(d)
                    drspeed = drone_speeds[d]
                    wavcen  = wav['wavelet_center']
                    expstarttime = wav['clock_time'] 

                    for p in remaining_packages:
                        packspeed = get_current_speed_of_package(p)
                        if packspeed < drspeed: # handoffs only happen to faster guys
                             packloc   = get_current_position_of_package(p)
                             tI, _     = get_interception_time_and_x_generalized(packloc, packspeed, 
                                                                                 wavcen, drspeed,
                                                                                 targets[p], global_clock_time, expstarttime)
                             if tI < nonm_tmin:
                                 nonm_tmin = tI - global_clock_time
                                 nonm_dmin = d
                                 nonm_pmin = p

          #print Fore.CYAN, "non-matched drones are ", non_matched_drones, Style.RESET_ALL
          print Fore.CYAN, "nonm_tmin: ", nonm_tmin, " ymin: ", ymin, " ewmin: ", ewmin, Style.RESET_ALL
          #-----------------------------------------------------------------------------------------------------------------        
          if nonm_tmin < min(ymin, ewmin) : # TYPE 0 EVENT (some nonmatched drone reaches package)
               print "TYPE 0 event detected"
               assert drone_locked_p[nonm_dmin] == False, ""
               time_till_event    = nonm_tmin
               global_clock_time += time_till_event
               
               # update the states of all the packages
               for p in remaining_packages:

                    sthat   = normalize(targets[p]-sources[p])
                    newposn = get_current_position_of_package(p) + time_till_event * get_current_speed_of_package(p)*sthat

                    if p != nonm_pmin:
                          old_handler = get_current_handler_of_package(p)
                          new_handler = old_handler
                    else :
                          print "old_handler: ",  
                          old_handler                 = get_current_handler_of_package(p)
                          new_handler                 = nonm_dmin
                          drone_locked_p[new_handler] = True

                          if old_handler is not None:
                              drone_pool.remove(old_handler)
                              

                          drone_wavelets_info[new_handler].append( {'wavelet_center'      : newposn,          \
                                                                    'clock_time'          : global_clock_time,\
                                                                    'matched_package_ids' : []}) 


                    package_trail_info[p].append({'current_position'  : newposn,\
                                                  'clock_time'        : global_clock_time,\
                                                  'current_handler_id': new_handler}) 

          #-----------------------------------------------------------------------------------------------------------------        
          elif ewmin < ymin:  # TYPE I EVENT (package reaches target)
              print "TYPE I event detected"
              #assert (pmin,dmin) not in [ d['edge_pair']  for d in lbend_edges_of_matching ], " "
              time_till_event    = ewmin
              global_clock_time += time_till_event

              # Process lbend edges in the matching for type \rnum{1} event
              for ledge in lbend_edges_of_matching:
                  (pl,dl) = ledge['edge_pair']
                  wav     = get_last_wavelet_of_drone(D[dl])     
                  wav['matched_package_ids'].append(P[pl])
              
              # Process straight edges in the matching for type \rnum{1} event
              package_delivered_p[P[pmin]] = True
              drone_locked_p[D[dmin]]      = False
              remaining_packages.remove(P[pmin])
              drone_pool.remove(D[dmin])

              for sedge in straight_edges_of_matching:
                    (ps, ds) = sedge
                    assert abs(get_current_speed_of_package(P[ps]) - drone_speeds[D[ds]]) < zerotol , "speeds should match"
                   
                    sthat = normalize(targets[P[ps]]-sources[P[ps]])
                    package_trail_info[P[ps]].append({'current_position'  : get_current_position_of_package(P[ps]) +\
                                                                          time_till_event * get_current_speed_of_package(P[ps]) * sthat,\
                                                   'clock_time'           : global_clock_time,\
                                                   'current_handler_id'   : ds}) 
                    wav = get_last_wavelet_of_drone(D[ds])     
                    wav['matched_package_ids'].append(P[ps])
              
              
          else:# TYPE II EVENT (a wavelet corresponding to a drone not handling 
               # a package reaches a package that might be stationary or being 
               # moved by another drone.)
              print "Type II event detected"
              #assert (plmin is not None and dlmin is not None), ""
              assert ymin < ewmin, ""
              time_till_event    = ymin
              global_clock_time += time_till_event

              # Process lbend edges in the matching for type \rnum{2} event
              print "lbend edges of matching ", lbend_edges_of_matching
              print "straight edges of matching ", straight_edges_of_matching

              for ledge in lbend_edges_of_matching:
                  (pl,dl) = ledge['edge_pair']
                  
                  sthat   = normalize(targets[P[pl]]-sources[P[pl]])
                  newposn = get_current_position_of_package(P[pl]) + time_till_event * get_current_speed_of_package(P[pl]) * sthat
                  package_trail_info[P[pl]].append({'current_position'   : newposn,\
                                                    'clock_time'         : global_clock_time, \
                                                    'current_handler_id' : get_current_handler_of_package(P[pl])   }) 
                  
                  wav     = get_last_wavelet_of_drone(D[dl])     
                  wav['matched_package_ids'].append(P[pl])

                  if pl == plmin and dl == dlmin:

                        if get_current_handler_of_package(P[plmin]) is not None:         
                            drone_locked_p[ get_current_handler_of_package(P[plmin])  ] = False
                            drone_pool.remove(get_current_handler_of_package(P[plmin]))

                        drone_locked_p[D[dlmin]] = True 
                        package_trail_info[P[pl]][-1]['current_handler_id'] = D[dlmin]
                        drone_wavelets_info[D[dlmin]].append( {'wavelet_center'      : newposn,          \
                                                               'clock_time'          : global_clock_time,\
                                                               'matched_package_ids' : []}) 
              
              # Process straight edges in the matching for type \rnum{2} event
              for sedge in straight_edges_of_matching:
                    (ps,ds) = sedge
                    assert (ps != plmin and ds != dlmin), ""
                    sthat   = normalize(targets[P[ps]]-sources[P[ps]])
                    package_trail_info[P[ps]].append({'current_position'  : get_current_position_of_package(P[ps]) +\
                                                                            time_till_event * get_current_speed_of_package(P[ps]) * sthat,\
                                                      'clock_time'        : global_clock_time,\
                                                      'current_handler_id': get_current_handler_of_package(P[ps])   }) 
                    wav = get_last_wavelet_of_drone(D[ds])     
                    wav['matched_package_ids'].append(P[ps])
              

     # Plot movement of packages and drones if \verb|plot_tour_p == True |
     
     fig, ax = plt.subplots()
     plot_tour_multiple_packages (fig, ax, "Multiple Package Handoff", 
               sources, targets, drone_initposns, drone_speeds, drone_wavelets_info, package_trail_info)
      
     return 
     #return pass pass pass pass pass 
 
def get_interception_time_and_x_generalized(s, us, p, up, t, c, k) :

    assert c-k>=0 , "c, global clock time should be greater than k,\
                    time when wavelet started expanding"

    zerotol = 1e-7
    if us < zerotol:
        l = np.linalg.norm(s-p)
        r = (c-k) * up
        if l >= r:
            return c + (l-r)/up, 0.0
        else:
            return np.inf, np.inf


    _ , x = get_interception_time_and_x(s,us,p,up,t,c-k)
 
    #### TODO! an assertion statement that makes sure that 
    #### c+x/us == k + |PM|/up where M is the meeting point 
    #### of the wavelets
    
    if x is not None:
         return c+x/us, x
    else:
         return np.inf, None

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


# distinct colors, obtained from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
        '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
        '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
        '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080'] 


numrobscanvas = 0
basespeed = 0.1

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
                 #speed    = np.random.uniform() # float(raw_input('What speed do you want for the drone at '+str(newPoint)))

                 global basespeed
                 run.drone_info.append( (newPoint, basespeed) ) 
                 patchSize  = (xlim[1]-xlim[0])/20.0
                 print Style.RESET_ALL
                 
                 global numrobscanvas
                 ax.add_patch( mpl.patches.Circle( newPoint, radius = patchSize,
                                                   facecolor='#EBEBEB', edgecolor='black'  ))

                 ax.text( newPoint[0], newPoint[1], "{:.2f}".format(basespeed) + "\n" + str(numrobscanvas) , fontsize=10, 
                          horizontalalignment='center', verticalalignment='center' )

                 numrobscanvas += 1    
                 ax.set_title('Number of drones inserted: ' +\
                              str(len(run.drone_info)), fontdict={'fontsize':25})
                 basespeed += 0.1

             elif event.button == 3:  
                 # Insert big colored circles representing the source and target points

                 patchSize  = (xlim[1]-xlim[0])/30.0
                 if (len(run.sources) + len(run.targets)) % 2 == 0 : 
                        run.sources.append((event.xdata, event.ydata))
                        ax.add_patch( mpl.patches.Circle( run.sources[-1], radius = patchSize, 
                                                   facecolor= cols[len(run.sources) % len(cols)], edgecolor='black', lw=1.0 ))
                        ax.text( run.sources[-1][0], run.sources[-1][1], 'S'+str(len(run.sources)-1), fontsize=15, 
                                 horizontalalignment='center', verticalalignment='center' )

                 else :
                      run.targets.append((event.xdata, event.ydata))
                      ax.add_patch( mpl.patches.Circle( run.targets[-1], radius = patchSize, 
                                                       facecolor= cols[len(run.sources)%len(cols)], edgecolor='black', lw=1.0 ))
                      ax.text( run.targets[-1][0], run.targets[-1][1], 'T'+str(len(run.targets)-1), fontsize=15, 
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
                            " (mm)     Match-and-move \n"                            +\
                            Style.RESET_ALL)

                    algo_str = algo_str.lstrip()
                     
                    # Incase there are patches present from the previous clustering, just clear them
                    clearAxPolygonPatches(ax)
                    if   algo_str == 'mm':
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

                    global numrobscanvas
                    numrobscanvas = 0
                    fig.canvas.draw()

                    global basespeed
                    basespeed = 0.1
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
   

def plot_tour_multiple_packages (fig, ax, figtitle, sources, targets, 
          drone_initposns, drone_speeds,
          drone_wavelets_info, 
          package_trail_info,
          xlims = [0,1],
          ylims = [0,1],
          aspect_ratio=1.0,
          speedfontsize=10,
          speedmarkersize=20,
          stmarkerfontsize=15,
          stmarkersize=20):
    
    import matplotlib.ticker as ticker
    ax.set_aspect(aspect_ratio)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    plt.rc('font', family='serif')
    ax.tick_params(which='both', top ='off', left='off',right='off', bottom ='off') 
    
    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.1', color='red')
    ax.grid(which='minor', linestyle=':', linewidth='0.1', color='black')

    stpatchSize  = (xlim[1]-xlim[0])/40.0
    drpatchSize  = 1.0 * stpatchSize

    # Draw the source, target, and initial positions of the robots as bold dots
    for source, target, stidx in zip(sources, targets, range(len(sources))):
         xs,ys = extract_coordinates([source, target])
         ax.plot(xs,ys, 'k--', alpha=0.3 ) # light line connecting source and target

         ax.add_patch( mpl.patches.Circle( source, radius = stpatchSize, facecolor= cols[stidx], edgecolor='black', lw=1.0, **{'alpha': 0.4}  ))
         ax.add_patch( mpl.patches.Circle( target, radius = stpatchSize, facecolor= cols[stidx], edgecolor='black', lw=1.0, **{'alpha': 0.4} ))

         ax.text(source[0], source[1], 'S'+str(stidx), fontsize =stmarkerfontsize, 
                 horizontalalignment ='center', verticalalignment   ='center')

         ax.text(target[0], target[1], 'T'+str(stidx), fontsize=stmarkerfontsize,\
                 horizontalalignment ='center', verticalalignment   ='center')

    # Draw speed labels on top of initial positions of the drones
    for idx in range(len(drone_initposns)):
         ax.add_patch( mpl.patches.Circle( (drone_initposns[idx][0], drone_initposns[idx][1]), 
                                           radius = drpatchSize, facecolor = 'gray', edgecolor = 'gray', lw=1.0,\
                                          **{'alpha': 0.4} ))

         ax.text( drone_initposns[idx][0], drone_initposns[idx][1], format(drone_speeds[idx],'.2f'),
                  fontsize = speedfontsize, horizontalalignment = 'center', verticalalignment   = 'center' )

    # Plot the trails of the packages (one color correponding to each package as in cols)
    for trail in package_trail_info:
         xs, ys = extract_coordinates([d['current_position'] for d in trail])
         print Fore.CYAN, "Printing Trail for package ", Style.RESET_ALL
         utils_algo.print_list(zip(xs,ys))
         plt.plot(xs,ys,'ro-')


    # Plot the paths of the drones (all drone paths have the same color, make them thickish and transparent)
    for dpath in drone_wavelets_info:
         xs, ys = extract_coordinates([ d['wavelet_center'] for d in dpath ])            
         print Fore.YELLOW, "Printing path of drone ", Style.RESET_ALL
         utils_algo.print_list(zip(xs,ys))
         plt.plot(xs,ys,'o-')

    plt.show()

