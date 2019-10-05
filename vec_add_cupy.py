#!/usr/bin/env python3
import cupy as cp
import numpy as np
import time

# Script for vec_add_cupy

def vec_add(a,b,n):
    xp = cp.get_array_module(a)
    c = xp.zeros(n)
    xp.add(a,b,c)
    return c

if __name__ == '__main__':
    debug = True
    n = 1 << 20
    num_iterations = 5
    a = np.random.uniform(1.0,10.0, n)
    b = np.random.uniform(5.0,15.0, n)
    diff = np.zeros(n)

    cpu_duration = 0.0
    for iteration in range(num_iterations):
        cpu_start_time = time.time()
        cpu_c = vec_add(a,b,n)
        cpu_duration += time.time() - cpu_start_time

    print("CPU Vector add of {} elements took {} seconds".format(n, cpu_duration/num_iterations))
    gpu_duration = 0.0
    with cp.cuda.Device(0):
        for iteration in range(num_iterations):
            gpu_start_time = time.time()
            a = cp.asarray(a)
            b = cp.asarray(b)
            c = vec_add(a,b,n)
            gpu_c = cp.asnumpy(c)
            gpu_duration += time.time() - gpu_start_time
        print("GPU Vector add of {} elements took {} seconds".format(n, gpu_duration/num_iterations))
 
        np.subtract(cpu_c, gpu_c, diff)
        if debug:
            print("There are {} differences in the cpu and gpu results".format(len(diff[diff>0.0])))
        if diff[diff>0.0]:
            error_index = np.argwhere(diff>3.0)
            print( "Error for element {}: cpu result was {} and gpu result was {}".format(error_index, 
                cpu_c[error_index], gpu_c[error_index]))


