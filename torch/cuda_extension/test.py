import torch
import torch.utils.benchmark as benchmark
import spherical_harmonics_cuda_extension

print (spherical_harmonics_cuda_extension.__file__)
import math
import numpy as np
import sphericart



def compute_spherical_harmonics_cuda(xyz, prefactors, requires_grad, dtype):
    xyz = xyz.type(dtype)
    prefactors = prefactors.type(dtype)

    '''
    void adjust_shared_memory(int lmax, int GRID_DIM_Y, bool requires_grad )

    everytime lmax, GRID_DIM_Y, requires_grad changes, adust_shared_memory needs to be called to:
    1. Check that we have sufficient shared memory for the buffers we're requesting
    2. Resize the shared memory allocation such that the buffers can fit. Atm it forces the resize even if the request fits in the existing shared memory allocation.

    this call is **expensive** - it should only be called in an __init__ function.

    note: can probably be greedy and just assume the gradients are always computed.
    '''

    spherical_harmonics_cuda_extension.adjust_shared_memory(xyz, lmax, 1, requires_grad)

    result = spherical_harmonics_cuda_extension._generic_spherical_harmonics_gpu(xyz, prefactors, lmax, 1)

    sph_harmonics = result[0]

    if (xyz.requires_grad):
        deriv = result[1]

        return sph_harmonics, deriv
    
    else:
        return sph_harmonics

def check_correctness(sph_harmonics_cuda,gradients_cuda,  sphericart_sph , sphericart_gradients):

    orig_dtype = sph_harmonics_cuda.dtype

    sph_harmonics_cuda = sph_harmonics_cuda.to(torch.float64)
    gradients_cuda = gradients_cuda.to(torch.float64)

    '''
    for performance reasons sph_harmonics and derivs are ordered as [nirreps, nedges] and [nirreps,3,nedges] on the GPU so need to transpose...
    '''
    #gradients = gradients_cuda.transpose(0,-1)
    #sph_harmonics = sph_harmonics_cuda.transpose(0,1)

    '''convert to FP64 to compare with CPU implementation'''
    sph_harmonics_cpu = sph_harmonics_cuda.cpu().detach().numpy()
    deriv_cpu = gradients_cuda.cpu().detach().numpy()

    '''
    lets test some correctness vs CPU implementation
    '''

    idx = np.where(sph_harmonics_cpu - sphericart_sph > 1e-5)

    if (len(idx[0]) > 1):
        print (f"diff spherical_harmonics in {orig_dtype}, check the following indices and values:")
        print (idx)
        print ('--CUDA--')
        print (sph_harmonics_cpu[idx])
        print ('--Reference--')
        print (sphericart_sph[idx])

    idx = np.where(deriv_cpu - sphericart_gradients > 5e-3)

    if (len(idx[0]) > 1):
        print (f"diff in gradients in {orig_dtype}, check the following indices and values:")
        print (idx)
        print ('--CUDA--')
        print (deriv_cpu[idx])
        print ('--Reference--')
        print (sh_sphericart_grad[idx])

def time_me(function, args):

    globs = {"func": function}

    globs.update(args)

    timer = benchmark.Timer(
        stmt='func(' + ", ".join(list(args.keys())) + ')',
        globals=globs)

    time = timer.timeit(1000)

    print (function)
    print (time)

nsample = 1000

xyz = torch.rand(nsample, 3, requires_grad=True).float().cuda()
xyz_cpu = xyz.cpu().double().detach().numpy()

for lmax in [0, 1, 5,7, 20, 25]:

    print ("-------", "lmax:", lmax, "-------")
    prefactors = torch.from_numpy(sphericart.wrappers.c_get_prefactors(lmax)).float().cuda()

    '''
    refernce CPU results in FP64
    '''
    sh_calculator = sphericart.SphericalHarmonics(lmax, normalized=True)
    sh_sphericart, sh_sphericart_grad = sh_calculator.compute(xyz_cpu, gradients=True)

    '''
    CUDA results in FP64
    '''
    sh_cuda, sh_cuda_deriv = compute_spherical_harmonics_cuda(xyz, prefactors, True, torch.float64)
    check_correctness(sh_cuda,sh_cuda_deriv, sh_sphericart, sh_sphericart_grad)

    '''
    CUDA results in FP32
    '''
    sh_cuda, sh_cuda_deriv = compute_spherical_harmonics_cuda(xyz, prefactors, True, torch.float32)
    check_correctness(sh_cuda,sh_cuda_deriv, sh_sphericart, sh_sphericart_grad)

    '''
    Timings for CPU and FP32 CUDA code
    '''
    time_me(sh_calculator.compute, {'xyz': xyz_cpu, 'gradients': True})
    
    spherical_harmonics_cuda_extension.adjust_shared_memory(xyz.float(), lmax, 1, True)
    time_me(spherical_harmonics_cuda_extension.spherical_harmonics_cuda, {'lmax': lmax, 'prefactors': prefactors.float(), 'xyz': xyz.float() })

    # spherical_harmonics_cuda_extension.adjust_shared_memory(xyz.double(), lmax, 1, True)
    # time_me(spherical_harmonics_cuda_extension.spherical_harmonics_cuda, {'lmax': lmax, 'xyz': xyz.double(), 'prefactors': prefactors.double()})