
import numpy as np
import random
from colorama import Fore
from colorama import Style

def vector_chain_from_point_list(pts):
    vec_chain = []
    for pair in zip(pts, pts[1:]):
        tail= np.array (pair[0])
        head= np.array (pair[1])
        vec_chain.append(head-tail)

    return vec_chain

def length_polygonal_chain(pts):
    vec_chain = vector_chain_from_point_list(pts)

    acc = 0
    for vec in vec_chain:
        acc = acc + np.linalg.norm(vec)
    return acc
def pointify_vector (x):
    if len(x) % 2 == 0:
        pts = []
        for i in range(len(x))[::2]:
            pts.append( [x[i],x[i+1]] )
        return pts
    else :
        sys.exit('List of items does not have an even length to be able to be pointifyed')
def flatten_list_of_lists(l):
       return [item for sublist in l for item in sublist]
def print_list(xs):
    for x in xs:
        print x
def partial_sums( xs ):
    psum = 0
    acc = []
    for x in xs:
        psum = psum+x
        acc.append( psum )
    return acc
def are_site_orderings_equal(sites1, sites2):

    for (x1,y1), (x2,y2) in zip(sites1, sites2): 
        if (x1-x2)**2 + (y1-y2)**2 > 1e-8:
            return False
    return True
def bunch_of_non_uniform_random_points(numpts):
    cluster_size = int(np.sqrt(numpts)) 
    numcenters   = cluster_size
    
    import scipy
    import random
    centers = scipy.rand(numcenters,2).tolist()

    scale, points = 4.0, []
    for c in centers:
        cx, cy = c[0], c[1]
        # For current center $c$ of this loop, generate \verb|cluster_size| points uniformly in a square centered at it
           
        sq_size      = min(cx,1-cx,cy, 1-cy)
        loc_pts_x    = np.random.uniform(low=cx-sq_size/scale, high=cx+sq_size/scale, size=(cluster_size,))
        loc_pts_y    = np.random.uniform(low=cy-sq_size/scale, high=cy+sq_size/scale, size=(cluster_size,))
        points.extend(zip(loc_pts_x, loc_pts_y))
        

    # Whatever number of points are left to be generated, generate them uniformly inside the unit-square
       
    num_remaining_pts = numpts - cluster_size * numcenters
    remaining_pts = scipy.rand(num_remaining_pts, 2).tolist()
    points.extend(remaining_pts)
    

    return points
    
def write_to_yaml_file(data, dir_name, file_name):
   import yaml
   with open(dir_name + '/' + file_name, 'w') as outfile:
     yaml.dump( data, outfile, default_flow_style = False)



def normalize(vec):
    vec = np.asarray(vec)
    return vec/np.linalg.norm(vec)
