'''

This is the file, written in Python, that
Parses the metric Jupyter notebook
Calculates the Christoffel Symbols
Writes and compiles the CUDA code
Calculates coordinate transformations (x, y, z) <-> (a, b, c)
Gets program parameters from input.json
Creates Surface instances
Runs LMGE_main file
Creates image from collisions and surface colors
Plots graph for particles

This file is ran from a command line

'''



# Python built-in imports
import os
import json
import math
import re
import csv

# Third-party imports
import nbformat as nb
import numpy as np
import sympy as sm
from sympy import symbols
from sympy.parsing.latex import parse_latex
from sympy.printing.cxxcode import cxxcode
from IPython.display import display

# Sympy differential geometry imports
from sympy.diffgeom import Manifold
from sympy.diffgeom import Patch
from sympy.diffgeom import CoordSystem
from sympy.diffgeom import metric_to_Christoffel_2nd
from sympy.diffgeom import TensorProduct as TP

# Matplotlib imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import cm



# If this is the main file
if __name__ == '__main__':
    
    # Initialize printing for debugging (not used in operation)
    sm.init_printing()
    
    # Create lorentzian manifold class
    M = Manifold('M', 4)
    
    # Create 4-dimensional patch on the manifold
    patch = Patch('P', M)
    
    # Create coordinate system with the coordinates (t, a, b, c)
    coord = CoordSystem('coord', patch, ['w', 'a', 'b', 'c'])
    
    # Gets coord functions and matching symbols used in parsing
    w, a, b, c = coord.coord_functions()
    W, A, B, C = sm.symbols('W A B C')
    
    # Gets first derivatives of coordinate functions
    oneforms = coord.base_oneforms()
    
    # Initializes arrays to store parsed data
    metric_parsed = np.empty((16), dtype=object)
    function_parsed = []
    limit_parsed = []
    error_parsed = []
    color_parsed = []
    type_parsed = []
    coord_parts = np.empty((8), dtype=object)
    
    # Open metric.ipynb as nbformat object
    input_notebook = nb.read('input\metric.ipynb', as_version=4)
    
    # Get source of notebook
    source = input_notebook['cells'][0]['source']
    
    # Iterate through number of characters in the source file
    for i in range(len(source)):
        
        # If Metric definition is found
        if source[i:i+8] == '# Metric':
            
            # Find all parts of the metric
            metric_parts = re.findall('\$\$(.*)\$\$', source)
            
            # Iterate through all metric parts
            i = 0
            for part in metric_parts:
                
                # Parse metric parts to sympy
                metric_parsed[i] = parse_latex(r'' + part[11:])
                i += 1
        
        # If Surface definitions are found
        if source[i:i+10] == '# Surfaces':
            
            # Get all parts of the surfaces
            function_parts = re.findall('\$ F = (.*) \$', source)
            limit_parts = re.findall('\$ L = (.*) \$', source)
            error_parts = re.findall('\$ E = (.*) \$', source)
            colorR_parts = re.findall('\$ Cr = (.*) \$', source)
            colorG_parts = re.findall('\$ Cg = (.*) \$', source)
            colorB_parts = re.findall('\$ Cb = (.*) \$', source)
            type_parts = re.findall('\$ T = (.*) \$', source)
            
            # Parse function parts to sympy
            for part in function_parts:
                function_parsed.append(parse_latex(r'' + part))
            
            # Parse limit parts to sympy
            for part in limit_parts:
                limit_parsed.append(parse_latex(r'' + part))
            
            # Parse error parts to sympy
            for part in error_parts:
                error_parsed.append(parse_latex(r'' + part))
                
            # Parse color parts to sympy
            for i in range(len(colorR_parts)):
                color_parsed.append(parse_latex(r'' + colorR_parts[i]))
                color_parsed.append(parse_latex(r'' + colorG_parts[i]))
                color_parsed.append(parse_latex(r'' + colorB_parts[i]))
            
            # Parse type parts to sympy
            for part in type_parts:
                type_parsed.append(parse_latex(r'' + part))
            
        
        # If Coordinate definitions are found
        if source[i:i+13] == '# Coordinates':
            
            # Get all parts of the coordinate functions
            coord_parts[0] = re.search('\$ t = (.*) \$', source).group(1)
            coord_parts[1] = re.search('\$ x = (.*) \$', source).group(1)
            coord_parts[2] = re.search('\$ y = (.*) \$', source).group(1)
            coord_parts[3] = re.search('\$ z = (.*) \$', source).group(1)
            coord_parts[4] = re.search('\$ w = (.*) \$', source).group(1)
            coord_parts[5] = re.search('\$ a = (.*) \$', source).group(1)
            coord_parts[6] = re.search('\$ b = (.*) \$', source).group(1)
            coord_parts[7] = re.search('\$ c = (.*) \$', source).group(1)
            
    # Create metric tensor using parsed data
    metric = sm.Matrix([[metric_parsed[0], metric_parsed[1], metric_parsed[2], metric_parsed[3]],
                        [metric_parsed[4], metric_parsed[5], metric_parsed[6], metric_parsed[7]],
                        [metric_parsed[8], metric_parsed[9], metric_parsed[10], metric_parsed[11]],
                        [metric_parsed[12], metric_parsed[13], metric_parsed[14], metric_parsed[15]]])
    
    # Convert metric tensor to line element using oneforms
    metric_new = sum([TP(di, dj)*metric[i, j] for i,di in enumerate(oneforms) for j,dj in enumerate(oneforms)]).subs({W:w,A:a,B:b,C:c})
    
    print('Getting Christoffel Symbols')
    
    # Calculate Christoffel Symbols from metric 
    christoffel = metric_to_Christoffel_2nd(metric_new)
    
    # Create CUDA coordinate symbols
    x_0, x_1, x_2, x_3 = sm.symbols('x_0 x_1 x_2 x_3')
    
    # Open CUDA file and read all lines to data
    with open('Geodesic\CUDA.cu', 'r') as cuda_file:
        data = cuda_file.readlines()
        cuda_file.close()
    
    # Initialize data_new list
    data_new = []
    
    # Iterate through all lines in data and append to data_new
    for line in range(len(data)):
        data_new.append(data[line])
    
    # Iterate through all lines in data_new and all values in the Christoffel Symbols
    for line in range(len(data_new)):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    
                    # If a Function definition is found
                    if data_new[line][0:117] == '__host__ __device__ float christoffel_{}{}{}(float x_0, float x_1, float x_2, float x_3) {{ return /* CHRISTOFFEL_{}{}{} */ '.format(i, j, k, i, j, k):
                        
                        # Add Christoffel Symbol definition into string after conversion to C++ code
                        data_new[line] = data_new[line][0:117] + cxxcode(christoffel[i, j, k].subs({w: x_0, a: x_1, b: x_2, c: x_3})) + ' /* CHRISTOFFEL_{}{}{} */; }}\n'.format(i, j, k)
    
    # If original data does not equal new data
    if data_new != data:
        # Open CUDA file
        with open('Geodesic\CUDA.cu', 'w') as cuda_file:
            print('Writing CUDA File')
            
            # Write all data to the CUDA file
            cuda_file.writelines(data_new)
            
            # Close CUDA file
            cuda_file.close()
    
            print('Compiling CUDA')
            
            # Compile Geodesic 
            os.system('msbuild LMGE\LMGE.sln /t:Build -p:Configuration=Release -p:Platform=x64')
    
    # Imports the LMGE function and Surface class
    from LMGE.LMGE_main import LMGE, Surface
    
    # Creates coordinate symbols
    # Program coordinates: (x, y, z) Derivative: (X, Y, Z)
    # Metric  coordinates: (a, b, c) Derivative: (A, B, C)
    s = sm.symbols('s')
    
    t = sm.Symbol('t', real=True)
    x = sm.Symbol('x', real=True)
    y = sm.Symbol('y', real=True)
    z = sm.Symbol('z', real=True)
    T = sm.Symbol('T', real=True)
    X = sm.Symbol('X', real=True)
    Y = sm.Symbol('Y', real=True)
    Z = sm.Symbol('Z', real=True)
    
    w = sm.Symbol('w', real=True)
    a = sm.Symbol('a', real=True)
    b = sm.Symbol('b', real=True)
    c = sm.Symbol('c', real=True)
    W = sm.Symbol('W', real=True)
    A = sm.Symbol('A', real=True)
    B = sm.Symbol('B', real=True)
    C = sm.Symbol('C', real=True)
    
    # Initializes arrays to store x and y expressions
    xexpr = np.empty((4), dtype=object)
    yexpr = np.empty((4), dtype=object)
    
    # Stores coordinate parts in their respective places
    xexpr[0] = coord_parts[0]
    xexpr[1] = coord_parts[1]
    xexpr[2] = coord_parts[2]
    xexpr[3] = coord_parts[3]
    
    yexpr[0] = coord_parts[4]
    yexpr[1] = coord_parts[5]
    yexpr[2] = coord_parts[6]
    yexpr[3] = coord_parts[7]
    
    # Initializes xe and ye arrays
    xe = np.empty((8), dtype=object)
    ye = np.empty((8), dtype=object)
    
    # Set values to parsed LaTeX expressions from xexpr and yexpr
    xe[0] = parse_latex(xexpr[0]).subs({sm.Symbol('w'): w, sm.Symbol('a'): a, sm.Symbol('b'): b, sm.Symbol('c'): c})
    xe[1] = parse_latex(xexpr[1]).subs({sm.Symbol('w'): w, sm.Symbol('a'): a, sm.Symbol('b'): b, sm.Symbol('c'): c})
    xe[2] = parse_latex(xexpr[2]).subs({sm.Symbol('w'): w, sm.Symbol('a'): a, sm.Symbol('b'): b, sm.Symbol('c'): c})
    xe[3] = parse_latex(xexpr[3]).subs({sm.Symbol('w'): w, sm.Symbol('a'): a, sm.Symbol('b'): b, sm.Symbol('c'): c})
    xe[4] = xe[0]
    xe[5] = xe[1]
    xe[6] = xe[2]
    xe[7] = xe[3]
    
    ye[0] = parse_latex(yexpr[0]).subs({sm.Symbol('t'): t, sm.Symbol('x'): x, sm.Symbol('y'): y, sm.Symbol('z'): z})
    ye[1] = parse_latex(yexpr[1]).subs({sm.Symbol('t'): t, sm.Symbol('x'): x, sm.Symbol('y'): y, sm.Symbol('z'): z})
    ye[2] = parse_latex(yexpr[2]).subs({sm.Symbol('t'): t, sm.Symbol('x'): x, sm.Symbol('y'): y, sm.Symbol('z'): z})
    ye[3] = parse_latex(yexpr[3]).subs({sm.Symbol('t'): t, sm.Symbol('x'): x, sm.Symbol('y'): y, sm.Symbol('z'): z})
    ye[4] = ye[0]
    ye[5] = ye[1]
    ye[6] = ye[2]
    ye[7] = ye[3]
    
    # Get derivatives of xe[4-7] and ye[4-7]
    for i in range(4):
        xe[4+i] = xe[4+i].diff(w) * W + xe[4+i].diff(a) * A + xe[4+i].diff(b) * B + xe[4+i].diff(c) * C
        ye[4+i] = ye[4+i].diff(t) * T + ye[4+i].diff(x) * X + ye[4+i].diff(y) * Y + ye[4+i].diff(z) * Z
    
    # Lambdify xe and ye
    for i in range(8):
        xe[i] = sm.lambdify((w, a, b, c, W, A, B, C), xe[i], 'numpy')
        ye[i] = sm.lambdify((t, x, y, z, T, X, Y, Z), ye[i], 'numpy')
    
    # Open settings.json file and load as a json object
    with open('input\settings.json', 'r') as settings_file:
        settings = json.load(settings_file)
    
    # Get all program parameters from settings object
    A = np.array(settings['Camera_Position'])
    B = np.array(settings['Camera_Pointing'])
    O = np.array(settings['Up_Vector'])
    
    a_bound = settings['a_bound']
    b_bound = settings['b_bound']
    nstep = settings['Geodesic_Solver_Steps']
    cstep = settings['Collision_Solver_Steps']
    tpb = settings['Threads_Per_Block']
    
    H = math.pi/180 * settings['Horziontal_FOV_Degrees']
    X_res = settings['Horizontal_Resolution']
    Y_res = settings['Vertical_Resolution']
    mode = settings['Mode']
    npar = settings['Number_Of_Particles']
    
    # Initialize surfaces array
    surfaces = np.empty((len(function_parsed)), dtype=object)
    
    # Iterate through surfaces
    for i in range(len(surfaces)):
        
        # Create Surface classes from parsed data
        F = function_parsed[i]
        L = float(limit_parsed[i])
        E = float(error_parsed[i])
        color = np.array([float(color_parsed[3*i+0]), float(color_parsed[3*i+1]), float(color_parsed[3*i+2])], dtype=np.float16)
        T = int(type_parsed[i])
        
        surfaces[i] = Surface(F, L, E, xe, cstep, T, color)
    
    # If in Ray-Tracing mode
    if mode == 0:

        # Sets number of particles equal to the number of pixels
        npar = X_res*Y_res

        # Gets verticle field of view and look vector
        V = 2*np.arctan(np.tan(H/2)*(Y_res/X_res))
        Z = B - A

        # Gets x and y camera basis vectors
        e_x = np.cross(Z,O)/np.linalg.norm(np.cross(Z,O))
        e_y = np.cross(e_x,Z)/np.linalg.norm(np.cross(e_x,Z))

        # Gets u and v vectors adjusted for field of view
        u = np.tan(H)*e_x
        v = np.tan(V)*e_y

        # Initializes initial values array
        F0 = np.zeros((X_res*Y_res, 8))

        # Iterates through X and Y resolution
        i = 0
        for s in range(X_res):
            for r in range(Y_res):

                # Gets relative pixel position to camera center
                m = (2*(s+0.5))/X_res - 1
                n = 1 - (2*(r+0.5))/Y_res

                # Gets vector pointing from camera center to pixel and normalizes
                P = m*u + n*v + Z/np.linalg.norm(Z)
                P = P/np.linalg.norm(P)

                # Sets initial values to respective values after transformation to metric coordinates (a, b, c)
                F0[i, 0] = ye[0](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 1] = ye[1](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 2] = ye[2](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 3] = ye[3](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 4] = ye[4](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 5] = ye[5](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 6] = ye[6](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])
                F0[i, 7] = ye[7](0, A[0], A[1], A[2], 1, P[0], P[1], P[2])

                print('Getting Inital Values: %d \r' % i, end='')

                i += 1

    # If in particle mode
    if mode == 1:

        # Sets number of particles variable to itself
        npar = npar

        # Initializes initial values array
        F0 = np.zeros((npar, 8))

        # Opens particles csv file
        particles_file = open(r'.\input\particles.csv', 'r')

        # Creates reader class for particles file
        particles = csv.reader(particles_file, delimiter=',')

        # Iterates through all rows in the file
        line_count = 0
        for row in particles:

            # If on the first row, skip
            if line_count == 0:
                line_count += 1

            # If on any other row
            else:
                print('Getting Inital Values: %d \r' % (line_count-1), end='')

                # Sets initial values to respective values after transformation to metric coordinates (a, b, c)
                F0[line_count-1, 0] = ye[0](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 1] = ye[1](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 2] = ye[2](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 3] = ye[3](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 4] = ye[4](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 5] = ye[5](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 6] = ye[6](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))
                F0[line_count-1, 7] = ye[7](float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), float(row[7]))

                line_count += 1

    print('')
    
    # Run LMGE function
    spline_out, col_out = LMGE(F0, xe, ye, a_bound, b_bound, nstep, cstep, npar, tpb, surfaces, mode)
    
    # If in ray-tracing mode
    if mode == 0:
        
        # Image save path
        image_save = 'output\out.png'
        
        # Get background color from settings object
        back_c = np.array(settings['Background_Color'], dtype=np.float16)
        
        # Initialize total color array
        total_color = np.zeros((X_res*Y_res, 3), dtype=np.float16)
        
        # Iterate through length of total_color
        for i in range(len(total_color)):
            
            # Set color equal to background color
            total_color[i, :] = back_c[:]
            
            # Iterate through surfaces
            for j in range(len(surfaces)):
                
                # If a collision is detected
                if col_out[i, j, 1] == True:
                    
                    # Set color equal to surface color
                    total_color[i, :] = surfaces[j].color[:]
        
        # Save image to path
        plt.imsave(image_save, np.rot90(total_color.reshape(Y_res, X_res, 3), 3))
    
    # If in particle mode
    if mode == 1:
        
        # Create figure to plot to and set bounds
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        
        # Iterate through number of particles
        for i in range(npar):
            
            # Plot in 3D the spline data
            ax.plot3D(spline_out[i, :, 1], spline_out[i, :, 2], spline_out[i, :, 3], 'r')
        
        # Show the plot
        tau = np.linspace(0, 2*np.pi, 200)
        plt.plot(1*np.cos(tau), 1*np.sin(tau), 'k-')
        plt.show()
        
        # Save the particle data
        np.savez('output\solved.npz', spline_out=spline_out, col_out=col_out)