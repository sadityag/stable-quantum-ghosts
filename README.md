To observe the naive quantized formulations of the stable quantum ghosts, there are 2 primary files:


1) exact_trajectories_schrodinger.py
2) 
  This file provides a static plot with the time coordinate indicated by a heatmap. The step size, grid size, etc are controllable parameters.

  Uses the Trotter decomposition to linear order in the time step to evolve the Hamiltonian.
  
  Solves a mixed-type PDE with a parabolic compact state space. Evaluates expectations of 2D position and momenta with the standard measure.
  
  Kinetic evolution partially done in Fourier space. Momenta are evaluated via FFTs.

  
3) schrodinger_trajectories_animated.py
   
  Animates into a gif and mp4 (requires ffmpeg).
  
  Provides an additional 2-D spatial state density plot.
  

Required libraries will be updated through a conda enviroment yaml specification in a future commit.
