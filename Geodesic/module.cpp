/*

This is the file, written in C++, that runs the CUDA code using inputs from python

Contains the functions:
PyObject* Geodesic

This file is ran directly from python and returns directly also

*/



// Python header include
#include <Python.h>

// General includes
#include <Windows.h>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

// Includes main C++ header
#include "module.h"



// Struct to store the S and X vectors and creates a type return
struct S_X
{
	std::vector<std::vector<float>> S;
	std::vector<std::vector<std::vector<float>>> X;
};

typedef struct S_X S_X_t;



// Defines main CUDA function for this file
S_X_t cuda_main(std::vector<std::vector<float>> F0, float a_bound, float b_bound, size_t nstep, int threadsPerBlock);



// Main Geodesic function that acts as a bridge between Python and CUDA
// Takes Python arguments, converts them to C++ types and passes to CUDA
// Gets return arrays from CUDA, converts to Python types, and returns
PyObject* Geodesic(PyObject* self, PyObject* args)
{
	// Initializes Python objects
	PyObject *F0_p, *bounds_p, *nstep_p, *tpb_p;

	// Gets python input from args tuple and returns NULL if none are found
	if (!PyArg_ParseTuple(args, "OOOO", &F0_p, &bounds_p, &nstep_p, &tpb_p))
	{
		return NULL;
	}

	// Converts Python types to nset, tpb, a_bound, and b_bound
	size_t nstep = PyLong_AsSize_t(nstep_p);

	long tpb = PyLong_AsLong(tpb_p);

	float a_bound = (float)PyFloat_AsDouble(PyTuple_GetItem(bounds_p, 0));
	float b_bound = (float)PyFloat_AsDouble(PyTuple_GetItem(bounds_p, 1));

	// Gets size of initial values
	ssize_t F0_len = PyList_Size(F0_p);

	// Initializes F0 vector
	std::vector<std::vector<float>> F0(F0_len, std::vector<float>(8));

	// Converts Python F0 to C++ types and stores in F0 vector
	for (int i = 0; i < F0_len; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			F0[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(F0_p, i), j));
		}
	}

	// Frees memory
	Py_DECREF(F0_p);
	Py_DECREF(bounds_p);
	Py_DECREF(nstep_p);
	Py_DECREF(tpb_p);

	// Creates struct that stores CUDA output arrays
	S_X_t SX;

	// Runs main CUDA function
	SX = cuda_main(F0, a_bound, b_bound, nstep, tpb);

	// Creates new Python lists
	PyObject* X_p = PyList_New(SX.X.size());
	PyObject* S_p = PyList_New(SX.S.size());
	
	// Makes S Python list 2-dimensional and X Python list 3-dimensional
	// Sets list values to the converted values of the C++ arrays S and X
	for (int i = 0; i < SX.X.size(); i++)
	{
		PyList_SetItem(X_p, i, PyList_New(nstep));
		PyList_SetItem(S_p, i, PyList_New(nstep));
		for (int j = 0; j < nstep; j++)
		{
			PyList_SetItem(PyList_GetItem(X_p, i), j, PyList_New(8));
			PyList_SetItem(PyList_GetItem(S_p, i), j, PyFloat_FromDouble((double)SX.S[i][j]));
			for (int k = 0; k < 8; k++)
			{
				PyList_SetItem(PyList_GetItem(PyList_GetItem(X_p, i), j), k, PyFloat_FromDouble((double)SX.X[i][j][k]));
			}
		}
	}

	// Creates return tuple and sets values equal to Python S and X arrays
	PyObject* SX_p = PyTuple_New(2); 

	PyTuple_SetItem(SX_p, 0, S_p);
	PyTuple_SetItem(SX_p, 1, X_p);

	// Returns tuple
	return SX_p;
}



// Creates method table so that Python can recognize the Geodesic function
static PyMethodDef Geodesic_methods[] = {
	{ "Geodesic", (PyCFunction)Geodesic, METH_VARARGS, nullptr },

	{ nullptr, nullptr, 0, nullptr }
};



// Creates module definition so that Python can recognize this library as a Python module
static PyModuleDef Geodesic_module = {
	PyModuleDef_HEAD_INIT,
	"Geodesic",
	"Implements geodesic equations and RKF45 method into C++ and CUDA",
	0,
	Geodesic_methods
};



// Initializes this library as a Python module
PyMODINIT_FUNC PyInit_Geodesic()
{
	return	PyModule_Create(&Geodesic_module);
}