#include <Python.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma once

#ifndef MODULE_H
#define MODULE_H

PyObject* Geodesic(PyObject* self, PyObject* args);

#endif