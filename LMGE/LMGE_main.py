'''

This is the file, written in Python, that 
Creates the initial values
Calls the Geodesic C++ library
Calculates holes
Calculates splines
Converts metric coordinates to rectangular
Calculates surfaces collisions

Contains the functions:
def getnan
def LMGE

Contains the class:
Surface: getHole(), getCollision()

This file is ran from the user Python file

'''



# Imports the Geodesic library
from .x64.Release.Geodesic import Geodesic

# Python built-in imports
import time
import csv
import cmath

# Third-party imports
import numpy as np
import scipy.interpolate as interp
from sympy import symbols
import sympy as sm



# Finds NaN's in the X array
# If one is found, the function returns the index and True
def getnan(S, X) -> (int, bool):

    # Checks for NaN or infinity in the X array
    finites = np.isfinite(X)

    # Iterates through the array
    for i in range(len(X)):
        for j in range(8):
            if finites[i, j] == False:
                return i, False
            else: continue

    return len(X) - 1, True



# Surface class
class Surface():

    # Initialization function
    def __init__(self, F, L, E, xe, cstep, type, color = np.array([1, 1, 1], dtype=np.float16)) -> None:

        # Lambdifies the surface function
        self.F = sm.lambdify((symbols('x y z')), F)

        # Sets all other class attributes
        self.L = L
        self.E = E
        self.xe = xe
        self.cstep = cstep
        self.type = type
        self.color = color

    # Gets holes in the geodesic's path as defined by the surface function
    def getHole(self, x, y, z) -> (int, bool):

        # Iterates through array
        for i in range(len(x)):

            # Checks if the substitution of the position values into the surface function is less than user-defined threshold
            if self.F(x[i], y[i], z[i]) <= self.L + self.E:

                # Return index of collision and True
                return i, True

            # Continue through loop
            else:
                continue

        # Returns max index and False if no collision is found
        return len(x)-1, False

    # Gets collisions in the geodesic's path, but are not spaces of undefined value
    def getCollision(self, x, y, z) -> (int, bool):

        # Iterates through a user-defined number of steps
        for i in range(self.cstep):

            # Checks if the substitution of the position values into the surface function is less than user-defined threshold
            if self.F(x[i], y[i], z[i]) <= self.L + self.E:

                # Return index of collision and True
                return i, True

            # Continue through loop
            else:
                continue

        # Returns 0 and True if no collision is found
        return 0, False



# LMGE function called from user Python file
# Calculates initial values
# Runs Geodesic library
# Calculates holes, spline, coordinate transformations, and collisions
def LMGE(F0, xe, ye, a_bound, b_bound, nstep, cstep, npar, tpb, surfaces, mode) -> (np.ndarray, np.ndarray):

    print('Starting GPU')

    # Runs Geodesic library with initial values, bounds, number of steps, and threads per block
    S, X = Geodesic(F0.tolist(), (a_bound, b_bound), nstep, tpb)
    print('')
    print('GPU Finished')

    # Initializes S and X arrays
    S = np.array(S)
    X = np.array(X)

    # Inserts initial values into the start of the arrays
    S = np.insert(S, 0, a_bound, axis=1)
    X = np.insert(X, 0, F0, axis=1)

    # Initializes empty arrays for temporary usage
    X_tmp = np.empty((len(X)), dtype=object)
    S_tmp = np.empty((len(S)), dtype=object)

    # Initializes temporary collisions and output collisions arrays
    col = np.empty((len(surfaces)), dtype=bool)
    col_out = np.empty((npar, len(surfaces), 2), dtype=object)

    x_val = np.zeros((npar, nstep))
    y_val = np.zeros((npar, nstep))
    z_val = np.zeros((npar, nstep))

    # Converts X array to rectangular coordinates
    for i in range(npar):
        for j in range(nstep):
            x_val[i, j] = xe[1](X[i, j, 0], X[i, j, 1], X[i, j, 2], X[i, j, 3], 0, 0, 0, 0)
            y_val[i, j] = xe[2](X[i, j, 0], X[i, j, 1], X[i, j, 2], X[i, j, 3], 0, 0, 0, 0)
            z_val[i, j] = xe[3](X[i, j, 0], X[i, j, 1], X[i, j, 2], X[i, j, 3], 0, 0, 0, 0)

        print('Changing Coordinates: %d \r' % i, end='')

    print('')

    # Iterates through number of particles
    for i in range(npar):

        # Initializes k and k_f
        k = nstep - 1
        k_f = nstep - 1

        # Iterates through surfaces
        for j in range(len(surfaces)):

            # If a hole-type surface
            if surfaces[j].type == 0:

                # Check for finites in X array
                k_f, finite = getnan(S[i], X[i])

                # Check for holes in the geodesic path
                k, col[j] = surfaces[j].getHole(x_val[i], y_val[i], z_val[i])

                # Set respective value in the output collisions array
                col_out[i, j, 0] = k
                col_out[i, j, 1] = col[j]

        # Set k to minimum of k and k_f
        k = min(k_f, k)

        # If there is a collision
        if True in col:

            # Restrict X_tmp and S_tmp to index k
            X_tmp[i] = X[i, 0:k, :]
            S_tmp[i] = S[i, 0:k]

        # If there is not a collision
        else:
            
            # No restictions are applied
            X_tmp[i] = X[i]
            S_tmp[i] = S[i]

        print('Getting Holes: %d \r' % i, end='')

    print('')

    # Initializes spline array
    spline = np.zeros((npar, 8), dtype=object)

    # Iterate through number of particles and 0-7
    for i in range(npar):
        for j in range(8):
            print('Getting Splines: %d \r' % i, end='')

            # Calculate splines from the temporary S and X arrays
            spline[i, j] = interp.interp1d(S_tmp[i], X_tmp[i][:,j], kind='cubic', copy = False, bounds_error = False, fill_value = X_tmp[i][-1, j])

    print('')

    # Free memory
    del X_tmp
    del S_tmp

    # Create linear space to calculate coordinate transformations from
    T = np.linspace(a_bound, b_bound, nstep)

    spline_tmp = np.empty((npar, nstep, 8))

    for i in range(npar):
        for j in range(8):
            spline_tmp[i, :, j] = spline[i, j](T)
        print('Getting Temporary Splines: %d \r' % i, end='')

    print('')

    # Initialize output spline array
    spline_out = np.empty((npar, nstep, 8), dtype=object)

    # Iterate through number of particles
    for i in range(npar):
        for j in range(nstep):
            for k in range(8):

                # Set respective values of output spline array to spline values transformed from metric coordinates (w, a, b, c) -> (t, x, y, z)
                x = xe[k](*spline_tmp[i, j, :].tolist())
                spline_out[i, j, k] = x.real
        
        print('Changing Coordinates: %d \r' % i, end='')

    print('')

    for i in range(npar):

        # Iterates through surfaces
        for j in range(len(surfaces)):

            # If normal-type surface
            if surfaces[j].type == 1:

                # Check for collisions in geodesic path
                col_out[i, j, 0], col_out[i, j, 1] = surfaces[j].getCollision(spline_out[i, :, 1], spline_out[i, :, 2], spline_out[i, :, 3])

        print('Getting Collisions: %d \r' % i, end='')

    print('')

    # Return output spline and collision arrays
    return (spline_out, col_out)