/*************************************************************************************
  ____            _                      _   _                 _        __  __ 
 |  _ \ __ _  ___| | ____ _  __ _  ___  | | | | __ _ _ __   __| | ___  / _|/ _|
 | |_) / _` |/ __| |/ / _` |/ _` |/ _ \ | |_| |/ _` | '_ \ / _` |/ _ \| |_| |_ 
 |  __/ (_| | (__|   < (_| | (_| |  __/ |  _  | (_| | | | | (_| | (_) |  _|  _|
 |_|   \__,_|\___|_|\_\__,_|\__, |\___| |_| |_|\__,_|_| |_|\__,_|\___/|_| |_|  
                            |___/                                              
  
  This package contains routines for rendering various aspects of the
  package handoff problems. This code is meant to be embedded as an appendix
  to the literate document containing implementations of approximation
  algorithms in Python 2.7

  The most important part for rendering package handoff problems is rendering
  *drones* and rendering the *site*, *target* pairs. Various meta-data tidbits have
  to be stuffed into the circles representing them.

  Further we need to be able to render the routes of the active drones.

  - The path of a single package is represented by that of a color.
  - The path of a single drone is indicated by that of a single dashing pattern.

  For each drone, we represent the velocity and fuel as a tuple, where both
  are supposed to be integers. The coordinates are assumed to be that of a back
  ground grid. Note that you should represent the speed and fuel as a binomial
  coefficient of the time $n \choose k$

  The active drones are labelled with letters: A,B,C,...
  The packages are labelled with numbers     : 1,2,3,4,5

  Colors used are R,G,B,Y,V
  
  Now note that all drones travel at maximum speed, or they might wait for a
  while. For the sake of legibility you might want to give units to fuel, speed
  and time, otherwise it will end up becoming a morass of numbers. 

  To give input we will give, two files
  1. *sourcetarget_info.csv*
  2. *drones_info.csv*
  3. *output.csv*

  https://tex.stackexchange.com/a/12710/17858 for drawing squiggly lines.


********************************************************************/
def parse_data_files(string sourcetarget_file,
		     string drones_file      ,
		     string outputfile)
{
   return info
}
/********************************************************************/
def render_site_target_pairs(picture pic = currentpicture,
			     pair[] sources              ,
			     pair[] targets)
{
  
    
}
/********************************************************************/
def render_drones(picture pic=currentpicture,
		  pair[] initdroneposns     ,
		  real[] speeds             ,
		  real[] fuel)
{
   
}
/********************************************************************/
def render_package_path(picture pic=currentpicture,
			path p, color pathcol     ,
			int[]  edge_drone_ids     ,
			real[] package_waiting_times)
{
  /*
   Define dictionary mapping package numbers to colors. Illustrations
   typically contain upto five packages at worst, so insert an error when
   this happens, so that you know to insert less than 5 packages.

   There might be multiple drones however, typically upto say 10,
   although this is more loosely given. 
  */
}

