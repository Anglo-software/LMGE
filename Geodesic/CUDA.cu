/*

This is the file, written in C++ that is compiled into GPU machine code

Contains the functions:

__global__ double* RK4
__host__ __device__ double* Equation
__host__ __device__ double* christoffel
Struct main_cuda

The main C++ library, Geodesic, calls this file to run the RK4 solving in parallel on the CUDA GPU

Python, at runtime, compiles this file if the christoffel symbols are edited

*/


// CUDA-specific includes
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

// General includes
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>

// Includes main C++ header
#include "module.h"



// 64 different functions that store the christoffel symbols across a 4x4x4 matrix
// These functions are written with comments so that python can find the specific ones
__host__ __device__ float christoffel_000(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_000 */ 0 /* CHRISTOFFEL_000 */; }
__host__ __device__ float christoffel_001(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_001 */ 0 /* CHRISTOFFEL_001 */; }
__host__ __device__ float christoffel_002(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_002 */ 0 /* CHRISTOFFEL_002 */; }
__host__ __device__ float christoffel_003(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_003 */ 0 /* CHRISTOFFEL_003 */; }
__host__ __device__ float christoffel_010(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_010 */ 0 /* CHRISTOFFEL_010 */; }
__host__ __device__ float christoffel_011(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_011 */ (1.0/2.0)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 5.0/3.0) /* CHRISTOFFEL_011 */; }
__host__ __device__ float christoffel_012(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_012 */ 0 /* CHRISTOFFEL_012 */; }
__host__ __device__ float christoffel_013(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_013 */ 0 /* CHRISTOFFEL_013 */; }
__host__ __device__ float christoffel_020(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_020 */ 0 /* CHRISTOFFEL_020 */; }
__host__ __device__ float christoffel_021(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_021 */ 0 /* CHRISTOFFEL_021 */; }
__host__ __device__ float christoffel_022(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_022 */ (3.0/8.0)*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) /* CHRISTOFFEL_022 */; }
__host__ __device__ float christoffel_023(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_023 */ 0 /* CHRISTOFFEL_023 */; }
__host__ __device__ float christoffel_030(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_030 */ 0 /* CHRISTOFFEL_030 */; }
__host__ __device__ float christoffel_031(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_031 */ 0 /* CHRISTOFFEL_031 */; }
__host__ __device__ float christoffel_032(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_032 */ 0 /* CHRISTOFFEL_032 */; }
__host__ __device__ float christoffel_033(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_033 */ (3.0/8.0)*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) /* CHRISTOFFEL_033 */; }
__host__ __device__ float christoffel_100(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_100 */ 0 /* CHRISTOFFEL_100 */; }
__host__ __device__ float christoffel_101(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_101 */ (1.0/2.0)/(-3.0/2.0*x_0 + (3.0/2.0)*x_1) /* CHRISTOFFEL_101 */; }
__host__ __device__ float christoffel_102(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_102 */ 0 /* CHRISTOFFEL_102 */; }
__host__ __device__ float christoffel_103(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_103 */ 0 /* CHRISTOFFEL_103 */; }
__host__ __device__ float christoffel_110(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_110 */ (1.0/2.0)/(-3.0/2.0*x_0 + (3.0/2.0)*x_1) /* CHRISTOFFEL_110 */; }
__host__ __device__ float christoffel_111(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_111 */ -1.0/2.0/(-3.0/2.0*x_0 + (3.0/2.0)*x_1) /* CHRISTOFFEL_111 */; }
__host__ __device__ float christoffel_112(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_112 */ 0 /* CHRISTOFFEL_112 */; }
__host__ __device__ float christoffel_113(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_113 */ 0 /* CHRISTOFFEL_113 */; }
__host__ __device__ float christoffel_120(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_120 */ 0 /* CHRISTOFFEL_120 */; }
__host__ __device__ float christoffel_121(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_121 */ 0 /* CHRISTOFFEL_121 */; }
__host__ __device__ float christoffel_122(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_122 */ -std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0)*(-3.0/8.0*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_122 */; }
__host__ __device__ float christoffel_123(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_123 */ 0 /* CHRISTOFFEL_123 */; }
__host__ __device__ float christoffel_130(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_130 */ 0 /* CHRISTOFFEL_130 */; }
__host__ __device__ float christoffel_131(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_131 */ 0 /* CHRISTOFFEL_131 */; }
__host__ __device__ float christoffel_132(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_132 */ 0 /* CHRISTOFFEL_132 */; }
__host__ __device__ float christoffel_133(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_133 */ -std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0)*(-3.0/8.0*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_133 */; }
__host__ __device__ float christoffel_200(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_200 */ 0 /* CHRISTOFFEL_200 */; }
__host__ __device__ float christoffel_201(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_201 */ 0 /* CHRISTOFFEL_201 */; }
__host__ __device__ float christoffel_202(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_202 */ (-3.0/8.0*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_202 */; }
__host__ __device__ float christoffel_203(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_203 */ 0 /* CHRISTOFFEL_203 */; }
__host__ __device__ float christoffel_210(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_210 */ 0 /* CHRISTOFFEL_210 */; }
__host__ __device__ float christoffel_211(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_211 */ 0 /* CHRISTOFFEL_211 */; }
__host__ __device__ float christoffel_212(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_212 */ ((3.0/8.0)*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_212 */; }
__host__ __device__ float christoffel_213(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_213 */ 0 /* CHRISTOFFEL_213 */; }
__host__ __device__ float christoffel_220(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_220 */ (-3.0/8.0*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_220 */; }
__host__ __device__ float christoffel_221(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_221 */ ((3.0/8.0)*x_0/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_221 */; }
__host__ __device__ float christoffel_222(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_222 */ 0 /* CHRISTOFFEL_222 */; }
__host__ __device__ float christoffel_223(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_223 */ 0 /* CHRISTOFFEL_223 */; }
__host__ __device__ float christoffel_230(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_230 */ 0 /* CHRISTOFFEL_230 */; }
__host__ __device__ float christoffel_231(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_231 */ 0 /* CHRISTOFFEL_231 */; }
__host__ __device__ float christoffel_232(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_232 */ 0 /* CHRISTOFFEL_232 */; }
__host__ __device__ float christoffel_233(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_233 */ (-3.0/2.0*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2) + (3.0/2.0)*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)) /* CHRISTOFFEL_233 */; }
__host__ __device__ float christoffel_300(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_300 */ 0 /* CHRISTOFFEL_300 */; }
__host__ __device__ float christoffel_301(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_301 */ 0 /* CHRISTOFFEL_301 */; }
__host__ __device__ float christoffel_302(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_302 */ 0 /* CHRISTOFFEL_302 */; }
__host__ __device__ float christoffel_303(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_303 */ (-3.0/8.0*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_303 */; }
__host__ __device__ float christoffel_310(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_310 */ 0 /* CHRISTOFFEL_310 */; }
__host__ __device__ float christoffel_311(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_311 */ 0 /* CHRISTOFFEL_311 */; }
__host__ __device__ float christoffel_312(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_312 */ 0 /* CHRISTOFFEL_312 */; }
__host__ __device__ float christoffel_313(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_313 */ ((3.0/8.0)*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_313 */; }
__host__ __device__ float christoffel_320(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_320 */ 0 /* CHRISTOFFEL_320 */; }
__host__ __device__ float christoffel_321(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_321 */ 0 /* CHRISTOFFEL_321 */; }
__host__ __device__ float christoffel_322(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_322 */ 0 /* CHRISTOFFEL_322 */; }
__host__ __device__ float christoffel_323(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_323 */ ((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_323 */; }
__host__ __device__ float christoffel_330(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_330 */ (-3.0/8.0*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/8.0)*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) + (3.0/4.0)*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_330 */; }
__host__ __device__ float christoffel_331(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_331 */ ((3.0/8.0)*x_0*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/8.0*x_1*std::pow(std::sin(x_2), 2)/std::pow(-3.0/2.0*x_0 + (3.0/2.0)*x_1, 2.0/3.0) - 3.0/4.0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_331 */; }
__host__ __device__ float christoffel_332(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_332 */ ((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::sin(x_2)*std::cos(x_2))/((3.0/2.0)*x_0*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2) - 3.0/2.0*x_1*std::cbrt(-3.0/2.0*x_0 + (3.0/2.0)*x_1)*std::pow(std::sin(x_2), 2)) /* CHRISTOFFEL_332 */; }
__host__ __device__ float christoffel_333(float x_0, float x_1, float x_2, float x_3) { return /* CHRISTOFFEL_333 */ 0 /* CHRISTOFFEL_333 */; }



// Main christoffel function that returns the others depending on the index parameters i, j, k
__host__ __device__ float christoffel(float x[4], int i, int j, int k)
{
    float x_0 = x[0];
    float x_1 = x[1];
    float x_2 = x[2];
    float x_3 = x[3];

    switch (i)
    {
    default: return 0;
    case 0: 
        switch (j)
        {
        default: return 0;
        case 0: 
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_000(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_001(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_002(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_003(x_0, x_1, x_2, x_3); break;
            }
        case 1: 
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_010(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_011(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_012(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_013(x_0, x_1, x_2, x_3); break;
            }
        case 2: 
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_020(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_021(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_022(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_023(x_0, x_1, x_2, x_3); break;
            }
        case 3: 
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_030(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_031(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_032(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_033(x_0, x_1, x_2, x_3); break;
            }
        }
    case 1:
        switch (j)
        {
        default: return 0;
        case 0:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_100(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_101(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_102(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_103(x_0, x_1, x_2, x_3); break;
            }
        case 1:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_110(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_111(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_112(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_113(x_0, x_1, x_2, x_3); break;
            }
        case 2:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_120(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_121(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_122(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_123(x_0, x_1, x_2, x_3); break;
            }
        case 3:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_130(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_131(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_132(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_133(x_0, x_1, x_2, x_3); break;
            }
        }
    case 2:
        switch (j)
        {
        default: return 0;
        case 0:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_200(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_201(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_202(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_203(x_0, x_1, x_2, x_3); break;
            }
        case 1:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_210(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_211(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_212(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_213(x_0, x_1, x_2, x_3); break;
            }
        case 2:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_220(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_221(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_222(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_223(x_0, x_1, x_2, x_3); break;
            }
        case 3:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_230(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_231(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_232(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_233(x_0, x_1, x_2, x_3); break;
            }
        }
    case 3:
        switch (j)
        {
        default: return 0;
        case 0:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_300(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_301(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_302(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_303(x_0, x_1, x_2, x_3); break;
            }
        case 1:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_310(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_311(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_312(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_313(x_0, x_1, x_2, x_3); break;
            }
        case 2:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_320(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_321(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_322(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_323(x_0, x_1, x_2, x_3); break;
            }
        case 3:
            switch (k)
            {
            default: return 0;
            case 0: return christoffel_330(x_0, x_1, x_2, x_3); break;
            case 1: return christoffel_331(x_0, x_1, x_2, x_3); break;
            case 2: return christoffel_332(x_0, x_1, x_2, x_3); break;
            case 3: return christoffel_333(x_0, x_1, x_2, x_3); break;
            }
        }
    }

    return 0;
}



// Main Equation function that takes input of the 4-position and 4-velocity and returns the results of the geodesic equations
__host__ __device__ float *Equation(float c_F0[8], float *c_dudv)
{
    // Initializes u, v, du, and dv
    float u[4] = { c_F0[0], c_F0[1], c_F0[2], c_F0[3] };
    float v[4] = { c_F0[4], c_F0[5], c_F0[6], c_F0[7] };
    float du[4] = { v[0], v[1], v[2], v[3] };
    float dv[4] = { 0, 0, 0, 0 };

    // Sets respective values of dv
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                dv[i] -= christoffel(u, i, j, k) * v[j] * v[k];
            }
        }
    }

    // Sets values of dudv
    c_dudv[0] = du[0];
    c_dudv[1] = du[1];
    c_dudv[2] = du[2];
    c_dudv[3] = du[3];
    c_dudv[4] = dv[0];
    c_dudv[5] = dv[1];
    c_dudv[6] = dv[2];
    c_dudv[7] = dv[3];

    // Returns resulting dudv
    return c_dudv;
}



// Main RK4 function
// This function is ran directly on the GPU and solves the geodesic equations using the specific inputs
// F0 - 4-position and 4-velocity
// dudv - pointer returned from the Equation function
// S - pointer returned by this function
// X - pointer returned by this function storing the solved data
// a_bound, b_bound - bounds of the equation
// nstep - number of steps to partition the bounds into
__global__ void RK4(float *c_F0, float *c_dudv, float* c_S, float* c_X, float a_bound, float b_bound, size_t nstep)
{
    // Gets offset of array indexes depending on thread and block number
    size_t offset = 8 * threadIdx.x + 8 * blockIdx.x * blockDim.x;

    // Gets starting s and step size
    float s = a_bound;
    float h = (b_bound - a_bound)/nstep;

    // Gets starting position and velocity
    float pos[4] = { c_F0[0 + offset], c_F0[1 + offset], c_F0[2 + offset], c_F0[3 + offset] };
    float vel[4] = { c_F0[4 + offset], c_F0[5 + offset], c_F0[6 + offset], c_F0[7 + offset] };

    // Initializes Y and Y_tmp and sets equal to starting pos and vel
    float Y[8];
    float Y_tmp[8];

    for (int i = 0; i < 8; i++)
    {
        Y[i] = c_F0[i + offset];
    }

    // Sets starting S and X value in the output arrays
    c_S[nstep * threadIdx.x + nstep * blockIdx.x * blockDim.x] = s;

    for (int i = 0; i < 8; i++)
    {
        c_X[i + offset * nstep] = c_F0[i + offset];
    }

    // Main RK4 for loop
    for (int step = 0; step < nstep; step++)
    {
        //Creates temporary arrays for the loop
        for (int i = 0; i < 4; i++)
        {
            Y[i] = pos[i];
            Y[i + 4] = vel[i];
        }

        __syncthreads();
        printf("Step number: %d \r", step);

        // Butcher Table for RK4:
        // 0.0 | 
        // 0.5 | 0.5
        // 0.5 | 0.0   0.5
        // 1.0 | 0.0   0.0   1.0
        //     +----------------------
        //       1/6   1/3   1/3   1/6
        // Defines the weights used for the method

        // Sets weights and evaluates k1
        for (int i = 0; i < 8; i++)
        {
            Y_tmp[i] = Y[i];
        }

        float* k1 = Equation(Y_tmp, c_dudv);

        // Sets weights and evaluates k2
        for (int i = 0; i < 8; i++)
        {
            Y_tmp[i] = Y[i] + 0.5 * h * k1[i];
        }

        float* k2 = Equation(Y_tmp, c_dudv);

        // Sets weights and evaluates k3
        for (int i = 0; i < 8; i++)
        {
            Y_tmp[i] = Y[i] + 0.5 * h * k2[i];
        }

        float* k3 = Equation(Y_tmp, c_dudv);

        // Sets weights and evaluates k4
        for (int i = 0; i < 8; i++)
        {
            Y_tmp[i] = Y[i] + 1.0 * h * k3[i];
        }

        float* k4 = Equation(Y_tmp, c_dudv);

        // Initializes delta to the weighted sum of the four k values
        float delta[8];

        for (int i = 0; i < 8; i++)
        {
            delta[i] = h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        // Adds delta to the position and velocity
        for (int i = 0; i < 4; i++)
        {
            pos[i] += delta[i];
            vel[i] += delta[i + 4];
        }

        // Sets next S value
        c_S[step + nstep * threadIdx.x + nstep * blockIdx.x * blockDim.x] = (step+1) * h;

        // Stores solved positions and velocities into the X array
        for (int i = 0; i < 4; i++)
        {
            c_X[i + 8 * step + offset * nstep] = pos[i];
            c_X[i + 4 + 8 * step + offset * nstep] = vel[i];
        }
    }
}



// Struct to store the S and X vectors and creates a type return
struct S_X
{
    std::vector<std::vector<float>> S;
    std::vector<std::vector<std::vector<float>>> X;
};

typedef struct S_X S_X_t;



// Main callable function for the cuda file
// F0 - Vector contatinging all starting pos and vel
// a_bound, b_bound - Bounds of solver
// nstep - Number of steps
// threadsPerBlock - Number of threads per blocks on the GPU grid   MAX: (GTX 1060: 1024, RTX 2070: 1024)
S_X_t cuda_main(std::vector<std::vector<float>> F0, float a_bound, float b_bound, size_t nstep, int threadsPerBlock)
{
    // Sets size and num variables, num is number of input values
    size_t size = 8 * sizeof(float);
    size_t num = F0.size();

    // Creates pointers to all host variables
    float* h_F0 = new float[8 * num];
    float* h_dudv = new float[8];
    float* h_S = new float[nstep * num];
    float* h_X = new float[8 * nstep * num];

    // Initializes host F0 to values of vector F0
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            h_F0[j + 8 * i] = F0[i][j];
        }
    }

    // Initializes all values of host dudv to 0
    for (int i = 0; i < 8; i++)
    {
        h_dudv[i] = 0;
    }

    // Creates cuda F0 and allocates to GPU memory
    float* c_F0 = nullptr;

    cudaMalloc((void**)&c_F0, size * num);

    // Creates cuda dudv and allocates to GPU memory
    float* c_dudv = nullptr;

    cudaMalloc((void**)&c_dudv, size);

    // Creates cuda S and allocates to GPU memory
    float* c_S = nullptr;

    cudaMalloc((void**)&c_S, nstep * num * sizeof(float));

    // Creates cuda X and allocates to GPU memory
    float* c_X = nullptr;

    cudaMalloc((void**)&c_X, size * nstep * num);

    // Copies values from host F0 to cuda F0
    cudaMemcpy(c_F0, h_F0, size * num, cudaMemcpyHostToDevice);

    // Calculates blocks per grid depending on number of threads and size of input
    int blocksPerGrid = num / threadsPerBlock;

    // Runs the RK4 method on the GPU
    RK4 <<<blocksPerGrid, threadsPerBlock>>> (c_F0, c_dudv, c_S, c_X, a_bound, b_bound, nstep);

    // Waits for all threads to finish
    cudaDeviceSynchronize();

    // Copies cuda dudv, S, and X to host dudv, S, and X
    cudaMemcpy(h_dudv, c_dudv, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_S, c_S, nstep * num * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_X, c_X, size * nstep * num, cudaMemcpyDeviceToHost);

    // Frees cuda F0, dudv, S, and X from GPU memory
    cudaFree(c_F0);
    cudaFree(c_dudv);
    cudaFree(c_S);
    cudaFree(c_X);

    // Creates S and X vectors
    std::vector<std::vector<float>> S(num, std::vector<float>(nstep));
    std::vector<std::vector<std::vector<float>>> X(num, std::vector<std::vector<float>>(nstep, std::vector<float>(8)));

    // Sets repective values for S and X vectors from host S and X arrays
    for (int i = 0; i < num; i++)
    {
        for (int j = 0; j < nstep; j++)
        {
            for (int k = 0; k < 8; k++)
            {
                X[i][j][k] = h_X[k + 8 * j + 8 * nstep * i];
            }
            S[i][j] = h_S[j + nstep * i];
        }
    }

    // Deletes host F0, dudv, S, and X arrays
    delete[] h_F0;
    delete[] h_dudv;
    delete[] h_S;
    delete[] h_X;

    // Creates output struct
    S_X_t SX;

    // Sets to repective vectors
    SX.S = S;
    SX.X = X;

    // Returns data
    return SX;
}