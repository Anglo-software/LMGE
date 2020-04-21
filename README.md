# LMGE (Lorentzian Manifold Geodesic Engine)

This program is a physics engine which is capable of calculating the paths of particles in curved space-time, such as near a dense body like a black hole. It is written in Python, C++, and CUDA to gain both high performance through GPU parallelization and use of common Python libraries like NumPy and MatPlotLib for ease of programming.

## How it works

WIP

## Current State of the Program

Currently, the program is fully operational with a few minor bugs.

## To-Do

* Implement all computation-intensive code to C++ or CUDA
* Provide way to input a discrete metric tensor, rather than only a continuous, symbolic one
* Implement multi-GPU support
* Optimize memory managment and parallelization in CUDA
* Create CPU-parallelized version
* Provide methods for multi-node servers for both GPU and CPU parallelized versions
