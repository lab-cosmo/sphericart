#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>

// #include <c10/util/Half.h>

#include "sphericart/cuda.hpp"

#define HARDCODED_LMAX 3
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_SAME_DTYPE(x, y) TORCH_CHECK(x.scalar_type() == y.scalar_type(), #x " and " #y " must have the same dtype.")

#define CHECK_INPUT(x)                                                                                                 \
    CHECK_CUDA(x);                                                                                                     \
    CHECK_CONTIGUOUS(x)

/*
Computes the index for buffer values which are threadIdx.x and threadIdx.y specific
*/
__device__ int get_index_(int i, int buff_size) { return threadIdx.y * buff_size + i * blockDim.x + threadIdx.x; }

/*
Computes the index for buffer values which are shared across GRID_DIM_Y
*/
__device__ int get_index(int i) { return i * blockDim.x + threadIdx.x; }
template <typename scalar_t> __device__ void compute_sph_l0(scalar_t *sph) {
    sph[get_index(0)] = 0.282094791773878;
}

template <typename scalar_t>
__device__ void compute_dsph_l0(scalar_t *sph_i, scalar_t *dxsph_i, scalar_t *dysph_i, scalar_t *dzsph_i) {
    dxsph_i[get_index(0)] = dysph_i[get_index(0)] = dzsph_i[get_index(0)] = 0.0;
}

template <typename scalar_t> __device__ void compute_sph_l1(scalar_t x, scalar_t y, scalar_t z, scalar_t *sph) {
    sph[get_index(1)] = 0.48860251190292 * y;
    sph[get_index(2)] = 0.48860251190292 * z;
    sph[get_index(3)] = 0.48860251190292 * x;
}

template <typename scalar_t>
__device__ void compute_dsph_l1(scalar_t *sph_i, scalar_t *dxsph_i, scalar_t *dysph_i, scalar_t *dzsph_i) {
    dxsph_i[get_index(1)] = 0.0;
    dxsph_i[get_index(2)] = 0.0;
    dxsph_i[get_index(3)] = 0.48860251190292;

    dysph_i[get_index(1)] = 0.48860251190292;
    dysph_i[get_index(2)] = 0.0;
    dysph_i[get_index(3)] = 0.0;

    dzsph_i[get_index(1)] = 0.0;
    dzsph_i[get_index(2)] = 0.48860251190292;
    dzsph_i[get_index(3)] = 0.0;
}

template <typename scalar_t>
__device__ void compute_sph_l2(scalar_t x, scalar_t y, scalar_t z, scalar_t x2, scalar_t y2, scalar_t z2, scalar_t *sph) {
    scalar_t tmp;
    tmp = 2.23606797749979 * x;
    sph[get_index(4)] = tmp * sph[get_index(1)];
    sph[get_index(7)] = tmp * sph[get_index(2)];
    sph[get_index(5)] = 2.23606797749979 * z * sph[get_index(1)];
    sph[get_index(6)] = -0.315391565252520 * (x2 + y2 - 2 * z2);
    sph[get_index(8)] = 0.54627421529604 * (x2 - y2);
}

template <typename scalar_t>
__device__ void compute_dsph_l2(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t *sph_i,
    scalar_t *dxsph_i,
    scalar_t *dysph_i,
    scalar_t *dzsph_i
) {
    dxsph_i[get_index(4)] = 2.23606797749979 * sph_i[get_index(1)];
    dxsph_i[get_index(5)] = 0.0;
    dxsph_i[get_index(6)] = -1.29099444873581 * sph_i[get_index(3)];
    dxsph_i[get_index(7)] = 2.23606797749979 * sph_i[get_index(2)];
    dxsph_i[get_index(8)] = 2.23606797749979 * sph_i[get_index(3)];

    dysph_i[get_index(4)] = -1.73205080756888 * dxsph_i[get_index(6)];
    dysph_i[get_index(5)] = dxsph_i[get_index(7)];
    dysph_i[get_index(6)] = -0.577350269189626 * dxsph_i[get_index(4)];
    dysph_i[get_index(7)] = 0.0;
    dysph_i[get_index(8)] = -dxsph_i[get_index(4)];

    dzsph_i[get_index(4)] = dzsph_i[get_index(8)] = 0.0;
    dzsph_i[get_index(5)] = dxsph_i[get_index(4)];
    dzsph_i[get_index(6)] = 1.15470053837925 * dxsph_i[get_index(7)];
    dzsph_i[get_index(7)] = dysph_i[get_index(4)];
}

template <typename scalar_t>
__device__ void compute_sph_l3(scalar_t x, scalar_t y, scalar_t z, scalar_t x2, scalar_t y2, scalar_t z2, scalar_t *sph) {
    scalar_t tmp;
    sph[get_index(9)] = -0.59004358992664 * y * (y2 - 3 * x2);
    sph[get_index(10)] = 2.64575131106459 * z * sph[get_index(4)];
    tmp = -0.457045799464466 * (x2 + y2 - 4 * z2);
    sph[get_index(11)] = y * tmp;
    sph[get_index(13)] = x * tmp;
    sph[get_index(12)] = -1.49270533036046 * z * (z2 - 2.37799637856361 * sph[get_index(6)]);
    sph[get_index(14)] = 1.44530572132028 * z * (x2 - y2);
    sph[get_index(15)] = 0.59004358992664 * x * (x2 - 3 * y2);
}

template <typename scalar_t>
__device__ void compute_dsph_l3(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t *sph_i,
    scalar_t *dxsph_i,
    scalar_t *dysph_i,
    scalar_t *dzsph_i
) {
    dxsph_i[get_index(9)] = 3.24037034920393 * sph_i[get_index(4)];
    dxsph_i[get_index(10)] = 2.64575131106459 * sph_i[get_index(5)];
    dxsph_i[get_index(11)] = -0.83666002653408 * sph_i[get_index(4)];
    dxsph_i[get_index(12)] = -2.04939015319192 * sph_i[get_index(7)];
    dxsph_i[get_index(13)] = 0.91409159892893 * (y2 - z2 + 4.75599275712721 * sph_i[get_index(6)]);
    dxsph_i[get_index(14)] = 2.64575131106459 * sph_i[get_index(7)];
    dxsph_i[get_index(15)] = 3.24037034920393 * sph_i[get_index(8)];

    dysph_i[get_index(9)] = dxsph_i[get_index(15)];
    dysph_i[get_index(10)] = dxsph_i[get_index(14)];
    dysph_i[get_index(11)] = -0.91409159892893 * (y2 - z2 - 1.58533091904240 * sph_i[get_index(6)]);
    dysph_i[get_index(12)] = -2.04939015319192 * sph_i[get_index(5)];
    dysph_i[get_index(13)] = -0.83666002653408 * sph_i[get_index(4)];
    dysph_i[get_index(14)] = -dxsph_i[get_index(10)];
    dysph_i[get_index(15)] = -dxsph_i[get_index(9)];

    dzsph_i[get_index(9)] = 0.0;
    dzsph_i[get_index(10)] = 2.64575131106459 * sph_i[get_index(4)];
    dzsph_i[get_index(11)] = 3.34664010613630 * sph_i[get_index(5)];
    dzsph_i[get_index(12)] = 3.54964786985977 * sph_i[get_index(6)];
    dzsph_i[get_index(13)] = 3.34664010613630 * sph_i[get_index(7)];
    dzsph_i[get_index(14)] = 2.64575131106459 * sph_i[get_index(8)];
    dzsph_i[get_index(15)] = 0.0;
}

template <typename scalar_t>
__device__ void compute_sph_l4(scalar_t x, scalar_t y, scalar_t z, scalar_t x2, scalar_t y2, scalar_t z2, scalar_t *sph) {
    scalar_t tmp;
    sph[get_index(16)] = 4.194391357527674 * sph[get_index(4)] * sph[get_index(8)];
    sph[get_index(17)] = 3 * z * sph[get_index(9)];
    tmp = -0.866025403784439 * (x2 + y2 - 6 * z2);
    sph[get_index(18)] = tmp * sph[get_index(4)];
    sph[get_index(22)] = tmp * sph[get_index(8)];
    sph[get_index(20)] = -0.69436507482941 * (
        y * sph[get_index(11)]
        - 1.6329931618554521 * z * sph[get_index(12)]
        + x * sph[get_index(13)]
    );
    tmp = -1.224744871391589 * (z2 - 4.755992757127213 * sph[get_index(6)]);
    sph[get_index(19)] = sph[get_index(5)] * tmp;
    sph[get_index(21)] = sph[get_index(7)] * tmp;
    sph[get_index(23)] = 3 * z * sph[get_index(15)];
    sph[get_index(24)] = -1.060660171779821 * (y * sph[get_index(9)] - x * sph[get_index(15)]);
}

template <typename scalar_t>
__device__ void compute_dsph_l4(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t *sph_i,
    scalar_t *dxsph_i,
    scalar_t *dysph_i,
    scalar_t *dzsph_i
) {
    dxsph_i[get_index(16)] = 4.242640687119285 * sph_i[get_index(9)];
    dxsph_i[get_index(17)] = 3.674234614174767 * sph_i[get_index(10)];
    dxsph_i[get_index(18)] = 1.892349391515120 * y * (y2 + 4.755992757127213 * sph_i[get_index(6)]);
    dxsph_i[get_index(19)] = -1.388730149658827 * sph_i[get_index(10)];
    dxsph_i[get_index(20)] = -2.777460299317654 * sph_i[get_index(13)];
    dxsph_i[get_index(21)] = -1.338093087114578 * (
        z * z2
        - 2.745873698591307 * y * sph_i[get_index(5)]
        - 4.019547514144073 * sph_i[get_index(12)]
    );
    dxsph_i[get_index(22)] = -1.892349391515120 * x * (x2 - 3 * z2);
    dxsph_i[get_index(23)] = 3.674234614174767 * sph_i[get_index(14)];
    dxsph_i[get_index(24)] = 4.242640687119285 * sph_i[get_index(15)];

    dysph_i[get_index(16)] = dxsph_i[get_index(24)];
    dysph_i[get_index(17)] = dxsph_i[get_index(23)];
    dysph_i[get_index(18)] = -1.892349391515120 * x * (y2 - 2 * z2 - 1.585330919042404 * sph_i[get_index(6)]);
    dysph_i[get_index(19)] = -1.338093087114578 * (z * (3 * y2 - z2) - 1.339849171381358 * sph_i[get_index(12)]);
    dysph_i[get_index(20)] = -2.777460299317654 * sph_i[get_index(11)];
    dysph_i[get_index(21)] = dxsph_i[get_index(19)];
    dysph_i[get_index(22)] = 1.892349391515120 * y * (y2 - 3 * z2);
    dysph_i[get_index(23)] = -dxsph_i[get_index(17)];
    dysph_i[get_index(24)] = -dxsph_i[get_index(16)];

    dzsph_i[get_index(16)] = 0.0;
    dzsph_i[get_index(17)] = 3 * sph_i[get_index(9)];
    dzsph_i[get_index(18)] = 3.927922024247863 * sph_i[get_index(10)];
    dzsph_i[get_index(19)] = 4.391550328268399 * sph_i[get_index(11)];
    dzsph_i[get_index(20)] = 4.535573676110727 * sph_i[get_index(12)];
    dzsph_i[get_index(21)] = 4.391550328268399 * sph_i[get_index(13)];
    dzsph_i[get_index(22)] = 3.927922024247863 * sph_i[get_index(14)];
    dzsph_i[get_index(23)] = 3 * sph_i[get_index(15)];
    dzsph_i[get_index(24)] = 0.0;
}

template <typename scalar_t>
__device__ void compute_sph_l5(scalar_t x, scalar_t y, scalar_t z, scalar_t x2, scalar_t y2, scalar_t z2, scalar_t *sph) {
    scalar_t tmp;
    sph[get_index(25)] = 13.12764113680340 * y * (y2 * (x2 - 0.2 * y2) + 0.3994658435740642 * sph[get_index(24)]);
    tmp = 3.316624790355400 * z;
    sph[get_index(26)] = tmp * sph[get_index(16)];
    sph[get_index(34)] = tmp * sph[get_index(24)];
    tmp = 4.974937185533100 * (z2 + 0.5284436396808015 * sph[get_index(6)]);
    sph[get_index(27)] = tmp * sph[get_index(9)];
    sph[get_index(33)] = tmp * sph[get_index(15)];
    tmp = 5.257947827012948 * sph[get_index(6)];
    sph[get_index(28)] = tmp * sph[get_index(10)];
    sph[get_index(32)] = tmp * sph[get_index(14)];
    tmp = 0.6324555320336759 * z;
    sph[get_index(29)] = 1.427248064296125 * (y * sph[get_index(20)] + tmp * sph[get_index(19)]);
    sph[get_index(31)] = 1.427248064296125 * (x * sph[get_index(20)] + tmp * sph[get_index(21)]);
    sph[get_index(30)] = 1.403403869441083 * (3.540173863740353 * sph[get_index(6)] * sph[get_index(12)] - z * z2 * z2);
    sph[get_index(35)] = -1.048808848170152 * (y * sph[get_index(16)] - x * sph[get_index(24)]);
}

template <typename scalar_t>
__device__ void compute_dsph_l5(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t *sph_i,
    scalar_t *dxsph_i,
    scalar_t *dysph_i,
    scalar_t *dzsph_i
) {
    dxsph_i[get_index(25)] = 5.244044240850758 * sph_i[get_index(16)];
    dxsph_i[get_index(26)] = 4.690415759823430 * sph_i[get_index(17)];
    dxsph_i[get_index(27)] =
        3.582364210034113 * (y2 * sph_i[get_index(4)] + 3.58568582800318 * x * sph_i[get_index(11)]);
    dxsph_i[get_index(28)] =
        -8.774964387392122 * ((y2 - z2) * sph_i[get_index(5)] + 0.3086066999241838 * sph_i[get_index(17)]);
    dxsph_i[get_index(29)] = -1.914854215512676 * sph_i[get_index(18)];
    dxsph_i[get_index(30)] = -3.496029493900505 * sph_i[get_index(21)];
    dxsph_i[get_index(31)] = -8.616843969807043 * (
        0.2102610435016800 * z2 * z2
        + 1.056887279361603 * sph_i[get_index(5)] * sph_i[get_index(5)]
        + (y2 - z2) * sph_i[get_index(6)] + 0.555555555555556 * sph_i[get_index(22)]
    );
    dxsph_i[get_index(32)] = -8.774964387392122 * (x2 - z2) * sph_i[get_index(7)];
    dxsph_i[get_index(33)] = -5.170697352496190 * (
        0.106904496764970 * z * dxsph_i[get_index(23)]
        - 0.320713490294909 * y * sph_i[get_index(9)]
        - sph_i[get_index(22)]
    );
    dxsph_i[get_index(34)] = 4.690415759823430 * sph_i[get_index(23)];
    dxsph_i[get_index(35)] = 5.24404424085076 * sph_i[get_index(24)];

    dysph_i[get_index(25)] = dxsph_i[get_index(35)];
    dysph_i[get_index(26)] = dxsph_i[get_index(34)];
    dysph_i[get_index(27)] = -3.102418411497714 * (
        0.534522483824849 * y * sph_i[get_index(9)]
        - 0.654653670707977 * z * sph_i[get_index(14)]
        - sph_i[get_index(22)]
    );
    dysph_i[get_index(28)] = -8.77496438739212 * (y2 - 1.585330919042404 * sph_i[get_index(6)]) * sph_i[get_index(7)];
    dysph_i[get_index(29)] = 0.7237468644557459 * (
        y * (2.12132034355964 * sph_i[get_index(9)]
        - 8.21583836257749 * sph_i[get_index(11)])
        +6.70820393249937 * z * sph_i[get_index(12)]
        + sph_i[get_index(24)]
    );
    dysph_i[get_index(30)] = -3.496029493900505 * sph_i[get_index(19)];
    dysph_i[get_index(31)] = dxsph_i[get_index(29)];
    dysph_i[get_index(32)] = 8.77496438739212 * (y2 - z2) * sph_i[get_index(5)];
    dysph_i[get_index(33)] = 3.582364210034113 * sph_i[get_index(4)] * (
        y2
        - 5 * z2
        - 1.585330919042404 * sph_i[get_index(6)]
    );
    dysph_i[get_index(34)] = -dxsph_i[get_index(26)];
    dysph_i[get_index(35)] = -dxsph_i[get_index(25)];

    dzsph_i[get_index(25)] = 0.0;
    dzsph_i[get_index(26)] = 3.316624790355400 * sph_i[get_index(16)];
    dzsph_i[get_index(27)] = 4.422166387140533 * sph_i[get_index(17)];
    dzsph_i[get_index(28)] = 5.066228051190221 * sph_i[get_index(18)];
    dzsph_i[get_index(29)] = 5.416025603090640 * sph_i[get_index(19)];
    dzsph_i[get_index(30)] = 5.527707983925666 * sph_i[get_index(20)];
    dzsph_i[get_index(31)] = 5.416025603090640 * sph_i[get_index(21)];
    dzsph_i[get_index(32)] = 5.066228051190221 * sph_i[get_index(22)];
    dzsph_i[get_index(33)] = 4.422166387140533 * sph_i[get_index(23)];
    dzsph_i[get_index(34)] = 3.316624790355400 * sph_i[get_index(24)];
    dzsph_i[get_index(35)] = 0.0;
}

template <typename scalar_t>
__device__ void compute_sph_l6(scalar_t x, scalar_t y, scalar_t z, scalar_t x2, scalar_t y2, scalar_t z2, scalar_t *sph) {
    scalar_t tmp;
    sph[get_index(36)] = 3.924637560539857 * sph[get_index(9)] * sph[get_index(15)];
    tmp = 3.605551275463989 * z;
    sph[get_index(37)] = tmp * sph[get_index(25)];
    sph[get_index(47)] = tmp * sph[get_index(35)];
    tmp = 6.4498061986388 * (z2 + 0.396332729760601 * sph[get_index(6)]);
    sph[get_index(38)] = tmp * sph[get_index(16)];
    sph[get_index(46)] = tmp * sph[get_index(24)];
    tmp = 1.04083299973307 * (z2 + 4.75599275712721 * sph[get_index(6)]);
    sph[get_index(39)] = tmp * sph[get_index(17)];
    sph[get_index(45)] = tmp * sph[get_index(23)];
    sph[get_index(40)] = 2.033805211017918 * (0.3779644730092272 * z * sph[get_index(28)] + x * sph[get_index(29)]);
    tmp = -6.399218702310463 * (z2 * z2 - 4.188790204786391 * sph[get_index(6)] * sph[get_index(6)]);
    sph[get_index(41)] = tmp * sph[get_index(5)];
    sph[get_index(43)] = tmp * sph[get_index(7)];
    sph[get_index(42)] = -1.087114613009218 * (
        0.645497224367903 * y * sph[get_index(29)]
        - z * sph[get_index(30)]
        + 0.645497224367903 * x * sph[get_index(31)]
    );
    sph[get_index(44)] = -0.9414688716912718 * (
        y * sph[get_index(27)]
        - 1.63299316185545 * z * sph[get_index(32)]
        + x * sph[get_index(33)]
    );
    sph[get_index(48)] = -1.040832999733066 * (y * sph[get_index(25)] - x * sph[get_index(35)]);
}

template <typename scalar_t>
__device__ void compute_dsph_l6(
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t x2,
    scalar_t y2,
    scalar_t z2,
    scalar_t *sph_i,
    scalar_t *dxsph_i,
    scalar_t *dysph_i,
    scalar_t *dzsph_i
) {
    scalar_t tmp;
    dxsph_i[get_index(36)] = 6.244997998398398 * sph_i[get_index(25)];
    dysph_i[get_index(48)] = -dxsph_i[get_index(36)];
    dxsph_i[get_index(37)] = 5.700877125495690 * sph_i[get_index(26)];
    dysph_i[get_index(47)] = -dxsph_i[get_index(37)];
    dxsph_i[get_index(38)] = -8.07303841165959 * y * (
        y2 * y2
        - 4.188790204786391 * sph_i[get_index(5)] * sph_i[get_index(5)]
        - 2.642218198404007 * sph_i[get_index(22)]
    );
    dxsph_i[get_index(39)] = -15.29705854077835 * (
        (y2 - z2) * sph_i[get_index(10)]
        + 0.2611164839335468 * sph_i[get_index(26)]
    );
    dxsph_i[get_index(40)] = 32.08092506951781 * (
        sph_i[get_index(5)] * (0.577350269189626 * y * sph_i[get_index(5)] - z * sph_i[get_index(6)])
        + 0.364182810197360 * y * y2 * sph_i[get_index(6)] + 0.3169804496925759 * sph_i[get_index(29)]
    );
    dxsph_i[get_index(41)] = -2.430862174021989 * sph_i[get_index(28)];
    dysph_i[get_index(43)] = dxsph_i[get_index(41)];
    dxsph_i[get_index(42)] = -4.210376791603422 * sph_i[get_index(31)];
    dysph_i[get_index(42)] = -4.210376791603422 * sph_i[get_index(29)];
    dxsph_i[get_index(43)] = 4.660970900149851 * (
        z2 * z * (1.666666666666667 * y2 + z2 - 2.642218198404007 * sph_i[get_index(6)])
        + 1.245553603643984 * y * sph_i[get_index(19)] + 1.781383145961857 * sph_i[get_index(30)]
    );
    dxsph_i[get_index(44)] = 14.73928415223878 * (x * (y2 - z2) * (2 * x2 - z2 - y2) + 0.2856568031469765 * sph_i[get_index(35)]);
    dxsph_i[get_index(45)] = 3.122498999199199 * (
        y * sph_i[get_index(17)]
        - 1.224744871391589 * z2 * sph_i[get_index(14)]
        + 1.846372364689991 * sph_i[get_index(32)]
    );
    tmp = 1.612451549659710 * (y * sph_i[get_index(16)] - 1.4142135623730950 * z * sph_i[get_index(23)]);
    dxsph_i[get_index(46)] = tmp + 6.18796485857095 * sph_i[get_index(33)];
    dysph_i[get_index(38)] = -tmp + 4.125309905713972 * sph_i[get_index(33)];
    dxsph_i[get_index(47)] = 5.700877125495690 * sph_i[get_index(34)];
    dxsph_i[get_index(48)] = 6.244997998398398 * sph_i[get_index(35)];
    dysph_i[get_index(36)] = dxsph_i[get_index(48)];
    dysph_i[get_index(37)] = dxsph_i[get_index(47)];
    dysph_i[get_index(39)] = -3.122498999199199 * (
        -1.22474487139159 * z2 * sph_i[get_index(14)]
        + y * sph_i[get_index(17)]
        - 1.10782341881399 * sph_i[get_index(32)]
    );
    dysph_i[get_index(40)] = 11.68332144554792 * (
        x * (-1.585330919042404 * sph_i[get_index(5)] * sph_i[get_index(5)] + (z2 - y2) * sph_i[get_index(6)])
        + 0.1740776559556978 * sph_i[get_index(31)]
    );
    dysph_i[get_index(41)] = -6.99145635022478 * z * (
        z2 * z2
        + (5.28443639680801 * y2 - 4.188790204786391 * sph_i[get_index(6)]) * sph_i[get_index(6)]
    );
    dysph_i[get_index(44)] = 13.49073756323204 * (
        y2 * z * sph_i[get_index(5)]
        + (-0.14940357616680 * x2 + 0.44821072850040 * y2 - 0.59761430466720 * z2) * sph_i[get_index(11)]
    );
    dysph_i[get_index(45)] = 7.648529270389177 * (y2 - z2 - 1.58533091904240 * sph_i[get_index(6)]) * sph_i[get_index(10)];
    dysph_i[get_index(46)] = 11.40175425099138 * (
        0.2360174359706574 * y2 * y2 * y
        + (y2 - 3 * z2) * sph_i[get_index(9)]
        + 0.1348399724926484 * sph_i[get_index(25)]
    );
    dzsph_i[get_index(36)] = 0.0;
    dzsph_i[get_index(37)] = 3.605551275463989 * sph_i[get_index(25)];
    dzsph_i[get_index(38)] = 4.861724348043977 * sph_i[get_index(26)];
    dzsph_i[get_index(39)] = 5.64881323014763 * sph_i[get_index(27)];
    dzsph_i[get_index(40)] = 6.14964891828646 * sph_i[get_index(28)];
    dzsph_i[get_index(41)] = 6.43145678393600 * sph_i[get_index(29)];
    dzsph_i[get_index(42)] = 6.52268767805531 * sph_i[get_index(30)];
    dzsph_i[get_index(43)] = 6.43145678393600 * sph_i[get_index(31)];
    dzsph_i[get_index(44)] = 6.14964891828646 * sph_i[get_index(32)];
    dzsph_i[get_index(45)] = 5.64881323014763 * sph_i[get_index(33)];
    dzsph_i[get_index(46)] = 4.861724348043977 * sph_i[get_index(34)];
    dzsph_i[get_index(47)] = 3.605551275463989 * sph_i[get_index(35)];
    dzsph_i[get_index(48)] = 0.0;
}

template <typename scalar_t>
__device__ void generic_sph_l_channel_device(
    int l,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t rxy,
    scalar_t twoz,
    scalar_t *sph,
    scalar_t *dsph_x,
    scalar_t *dsph_y,
    scalar_t *dsph_z,
    int sph_offset,
    scalar_t *pk,
    scalar_t *qlmk,
    scalar_t *c,
    scalar_t *s,
    bool requires_grad
) {
    scalar_t qlm_2, qlm_1, qlm_0;
    scalar_t ql1m_2, ql1m_1, ql1m_0;

    qlm_2 = qlmk[l];

    scalar_t pq = qlm_2 * pk[l];
    scalar_t pdq = 0.0;
    scalar_t pdqx = 0.0;
    scalar_t pdqy = 0.0;

    scalar_t s_l = s[get_index(l)];
    scalar_t s_l_neg1 = s[get_index(l - 1)];
    scalar_t c_l = c[get_index(l)];
    scalar_t c_l_neg1 = c[get_index(l - 1)];

    sph[get_index(sph_offset - l)] = pq * s_l;
    sph[get_index(sph_offset + l)] = pq * c_l;

    if (requires_grad) {
        pq *= l;
        dsph_x[get_index(sph_offset - l)] = pq * s_l_neg1;
        dsph_y[get_index(sph_offset - l)] = dsph_x[get_index(sph_offset + l)] = pq * c_l_neg1;
        dsph_y[get_index(sph_offset + l)] = -dsph_x[get_index(sph_offset - l)];
        dsph_z[get_index(sph_offset - l)] = 0;
        dsph_z[get_index(sph_offset + l)] = 0;
        ql1m_2 = 0;
    }

    qlm_1 = -z * qlm_2;
    pq = qlm_1 * pk[l - 1];
    sph[get_index(sph_offset - l + 1)] = pq * s_l_neg1;
    sph[get_index(sph_offset + l - 1)] = pq * c_l_neg1;

    if (requires_grad) {
        pq *= (l - 1);
        dsph_x[get_index(sph_offset - l + 1)] = pq * s[get_index(l - 2)];
        dsph_y[get_index(sph_offset + -l + 1)] = dsph_x[get_index(sph_offset + l - 1)] = pq * c[get_index(l - 2)];
        dsph_y[get_index(sph_offset + l - 1)] = -dsph_x[get_index(sph_offset - l + 1)];

        // uses Q(l-1)(l-1) to initialize the other recursion
        ql1m_1 = qlmk[-1];
        pdq = pk[l - 1] * (l + l - 1) * ql1m_1;
        dsph_z[get_index(sph_offset - l + 1)] = pdq * s_l_neg1;
        dsph_z[get_index(sph_offset + l - 1)] = pdq * c[get_index(l - 1)];
    }

    // and now do the other m's, decrementally
    auto twomz = l * twoz; // compute decrementally to hold 2(m+1)z
    for (auto m = l - 2; m > HARDCODED_LMAX - 1; --m) {
        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];

        auto s_m = s[get_index(m)];
        auto c_m = c[get_index(m)];

        auto s_m_neg1 = s[get_index(m - 1)];
        auto c_m_neg1 = c[get_index(m - 1)];

        sph[get_index(sph_offset - m)] = pq * s_m;
        sph[get_index(sph_offset + m)] = pq * c_m;

        if (requires_grad) {
            pq *= m;
            ql1m_0 = qlmk[m - l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1;
            ql1m_1 = ql1m_0; // shift

            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dsph_x[get_index(sph_offset - m)] = pdqx * s_m + pq * s_m_neg1;
            dsph_x[get_index(sph_offset + m)] = pdqx * c_m + pq * c_m_neg1;
            pdqy = pdq * y;
            dsph_y[get_index(sph_offset - m)] = pdqy * s_m + pq * c_m_neg1;
            dsph_y[get_index(sph_offset + m)] = pdqy * c_m - pq * s_m_neg1;
            pdq = pk[m] * (l + m) * ql1m_1;
            dsph_z[get_index(sph_offset - m)] = pdq * s_m;
            dsph_z[get_index(sph_offset + m)] = pdq * c_m;
        }
    }

    for (auto m = HARDCODED_LMAX - 1; m > 0; --m) {
        auto s_m = s[get_index(m)];
        auto c_m = c[get_index(m)];

        auto s_m_neg1 = s[get_index(m - 1)];
        auto c_m_neg1 = c[get_index(m - 1)];

        twomz -= twoz;
        qlm_0 = qlmk[m] * (twomz * qlm_1 + rxy * qlm_2);
        qlm_2 = qlm_1;
        qlm_1 = qlm_0; // shift

        pq = qlm_0 * pk[m];
        sph[get_index(sph_offset - m)] = pq * s_m;
        sph[get_index(sph_offset + m)] = pq * c_m;

        if (requires_grad) {
            pq *= m;
            ql1m_0 = qlmk[m - l] * (twomz * ql1m_1 + rxy * ql1m_2);
            ql1m_2 = ql1m_1;
            ql1m_1 = ql1m_0; // shift

            pdq = pk[m] * ql1m_2;
            pdqx = pdq * x;
            dsph_x[get_index(sph_offset - m)] = pdqx * s_m + pq * s_m_neg1;
            dsph_x[get_index(sph_offset + m)] = pdqx * c_m + pq * c_m_neg1;
            pdqy = pdq * y;
            dsph_y[get_index(sph_offset - m)] = pdqy * s_m + pq * c_m_neg1;
            dsph_y[get_index(sph_offset + m)] = pdqy * c_m - pq * s_m_neg1;
            pdq = pk[m] * (l + m) * ql1m_1;
            dsph_z[get_index(sph_offset - m)] = pdq * s_m;
            dsph_z[get_index(sph_offset + m)] = pdq * c_m;
        }
    }

    // m=0
    qlm_0 = qlmk[0] * (twoz * qlm_1 + rxy * qlm_2);
    sph[get_index(sph_offset)] = qlm_0 * pk[0];

    if (requires_grad) {
        ql1m_0 = qlmk[-l] * (twoz * ql1m_1 + rxy * ql1m_2);
        ql1m_2 = ql1m_1;
        ql1m_1 = ql1m_0; // shift
        // derivatives
        dsph_x[get_index(sph_offset)] = pk[0] * x * ql1m_2;
        dsph_y[get_index(sph_offset)] = pk[0] * y * ql1m_2;
        dsph_z[get_index(sph_offset)] = pk[0] * l * ql1m_1;
    }
}

template <typename scalar_t>
__device__ inline void clear_buffers(
    int nelements,
    scalar_t *sph,
    scalar_t *dsph_x,
    scalar_t *dsph_y,
    scalar_t *dsph_z,
    bool requires_grad
) {
    for (int i = 0; i < nelements; i++) {
        sph[get_index(i)] = 0.0;

        if (requires_grad) {
            dsph_x[get_index(i)] = 0.0;
            dsph_y[get_index(i)] = 0.0;
            dsph_z[get_index(i)] = 0.0;
        }
    }
    __syncthreads();
}

template <typename scalar_t>
__device__ inline void write_buffers(
    int atom_idx,
    int natoms,
    scalar_t x,
    scalar_t y,
    scalar_t z,
    scalar_t ir,
    int n_elements,
    int offset,
    scalar_t *buffer_sph,
    scalar_t *buffer_dsph_x,
    scalar_t *buffer_dsph_y,
    scalar_t *buffer_dsph_z,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph,
    bool requires_grad,
    bool normalize
) {
    if (atom_idx < natoms) {
        for (int i = 0; i < n_elements; i++) {
            sph[atom_idx][offset + i] = buffer_sph[get_index(i)];

            if (requires_grad) {
                auto tmp_dx = buffer_dsph_x[get_index(i)];
                auto tmp_dy = buffer_dsph_y[get_index(i)];
                auto tmp_dz = buffer_dsph_z[get_index(i)];

                // corrects derivatives for normalization
                if (normalize) {
                    auto tmp = (tmp_dx * x + tmp_dy * y + tmp_dz * z);

                    tmp_dx = (tmp_dx - x * tmp) * ir;
                    tmp_dy = (tmp_dy - y * tmp) * ir;
                    tmp_dz = (tmp_dz - z * tmp) * ir;
                }

                dsph[atom_idx][0][offset + i] = tmp_dx;
                dsph[atom_idx][1][offset + i] = tmp_dy;
                dsph[atom_idx][2][offset + i]= tmp_dz;
            }
        }
    }
}

template <typename scalar_t>
__global__ void spherical_harmonics_kernel(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> prefactors,
    int lmax,
    bool requires_grad,
    bool normalize,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph
) {
    extern __shared__ char buffer[];

    size_t offset = 0;

    scalar_t *buffer_c = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_s = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.x * (lmax + 1) * sizeof(scalar_t);
    scalar_t *buffer_prefactors = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += prefactors.size(0) * sizeof(scalar_t);

    //int nl = 2 * lmax + 1;

    int nl = max(
        static_cast<int>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * lmax + 1
     );

    scalar_t *buffer_sph = reinterpret_cast<scalar_t *>(buffer + offset);
    offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);

    scalar_t *buffer_dsph_x;
    scalar_t *buffer_dsph_y;
    scalar_t *buffer_dsph_z;

    if (requires_grad) {
        buffer_dsph_x = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
        buffer_dsph_y = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
        buffer_dsph_z = reinterpret_cast<scalar_t *>(buffer + offset);
        offset += blockDim.y * blockDim.x * nl * sizeof(scalar_t);
    }

    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int natoms = xyz.size(0);

    scalar_t x = 0.0;
    scalar_t y = 0.0;
    scalar_t z = 0.0;

    scalar_t x2 = 0.0;
    scalar_t y2 = 0.0;
    scalar_t z2 = 0.0;

    for (int i = threadIdx.x; i < prefactors.size(0); i += blockDim.x) {
        buffer_prefactors[i] = prefactors[i];
    }

    __syncthreads();

    if (atom_idx < natoms) {
        x = xyz[atom_idx][0];
        y = xyz[atom_idx][1];
        z = xyz[atom_idx][2];

        x2 = x * x;
        y2 = y * y;
        z2 = z * z;
    }

    scalar_t ir = 0.0;

    if (normalize) {
        if (atom_idx < natoms) {
            auto ir2 = 1.0 / (x2 + y2 + z2);
            ir = sqrt(ir2);
            x *= ir;
            y *= ir;
            z *= ir;
            x2 *= ir2;
            y2 *= ir2;
            z2 *= ir2;
        }
    }

    auto twoz = 2 * z;
    auto rxy = x2 + y2;

    if (threadIdx.y == 0) {
        buffer_c[get_index(0)] = 1.0;
        buffer_s[get_index(0)] = 0.0;

        for (int m = 1; m < lmax + 1; m++) {
            int m_in_idx = get_index(m - 1);
            int m_out_idx = get_index(m);

            scalar_t c = buffer_c[m_in_idx];
            scalar_t s = buffer_s[m_in_idx];

            buffer_c[m_out_idx] = c * x - s * y;
            buffer_s[m_out_idx] = c * y + s * x;
        }
    }

    __syncthreads();

    // work through hardcoded parts first...

    clear_buffers(
        (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1),
        buffer_sph,
        buffer_dsph_x,
        buffer_dsph_y,
        buffer_dsph_z,
        requires_grad
    ); 

    if (lmax >= 0) {

        //clear_buffers(1, buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z, requires_grad);

        compute_sph_l0(buffer_sph);

        if (requires_grad) {
            compute_dsph_l0(buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z);
        }
    }

    if (lmax >= 1) {

        compute_sph_l1(x, y, z, buffer_sph);

        if (requires_grad) {
            compute_dsph_l1(buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z);
        }
    }

    if (lmax >= 2) {

        compute_sph_l2(x, y, z, x2, y2, z2, buffer_sph);
        if (requires_grad) {
            compute_dsph_l2(x, y, z, x2, y2, z2, buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z);
        }
    }

    if (lmax >= 3) {

        compute_sph_l3(x, y, z, x2, y2, z2, buffer_sph);
        if (requires_grad) {
            compute_dsph_l3(x, y, z, x2, y2, z2, buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z);
        }
    }
    __syncthreads();

    write_buffers(
        atom_idx,
        natoms,
        x,
        y,
        z,
        ir,
        (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1),
        0,
        buffer_sph,
        buffer_dsph_x,
        buffer_dsph_y,
        buffer_dsph_z,
        sph,
        dsph,
        requires_grad,
        normalize
    );

    // now lets do the generic terms...
    int size_q = (lmax + 1) * (lmax + 2) / 2;
    int k = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 2) / 2;

    scalar_t *qlmk = buffer_prefactors + size_q + k;

    scalar_t *pk = buffer_prefactors + k;

    int base_index = (HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1);

    for (int l = HARDCODED_LMAX + 1; l < lmax + 1; l += 1) {
        int sph_offset = l; // sph needs to point to Y[l, 0]

        // sph 0 : 0
        // sph 1: 0 1 2
        // sph 2: 0 1 2 3 4
        // sph 3: 0 1 2 3 4 5 6

        // clear out temporary storage buffers
        clear_buffers(2 * l + 1, buffer_sph, buffer_dsph_x, buffer_dsph_y, buffer_dsph_z, requires_grad);

        // do some work
        generic_sph_l_channel_device(
            l,
            x,
            y,
            z,
            rxy,
            twoz,
            buffer_sph,
            buffer_dsph_x,
            buffer_dsph_y,
            buffer_dsph_z,
            sph_offset,
            pk,
            qlmk,
            buffer_c,
            buffer_s,
            requires_grad
        );

        // write out temporary storage buffers
        write_buffers(
            atom_idx,
            natoms,
            x,
            y,
            z,
            ir,
            2 * l + 1,
            base_index,
            buffer_sph,
            buffer_dsph_x,
            buffer_dsph_y,
            buffer_dsph_z,
            sph,
            dsph,
            requires_grad,
            normalize
        );

        base_index += 2 * l + 1;
        qlmk += l + 1;
        pk += l + 1;
    }
}


static size_t total_buffer_size(size_t l_max, size_t GRID_DIM_X, size_t GRID_DIM_Y, size_t dtype_size, bool requires_grad) {

    int nl = max(
        static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * l_max + 1
     );

    size_t total_buff_size = 0;

    total_buff_size += GRID_DIM_X * (l_max + 1) * dtype_size;      // buffer_c
    total_buff_size += GRID_DIM_X * (l_max + 1) * dtype_size;      // buffer_s
    total_buff_size += (l_max + 1) * (l_max + 2) * dtype_size;     // buffer_prefactors
    total_buff_size += GRID_DIM_Y * GRID_DIM_X * nl * dtype_size;  // buffer_sph_out

    if (requires_grad) {
        total_buff_size += 3 * GRID_DIM_Y * GRID_DIM_X * nl * dtype_size; // buffer_sph_derivs
    }

    return total_buff_size;
}

bool sphericart_torch::adjust_cuda_shared_memory(torch::ScalarType scalar_type, int64_t l_max, int64_t GRID_DIM_X, int64_t GRID_DIM_Y, bool requires_grad) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    size_t dtype = torch::elementSize(scalar_type);
    auto required_buff_size = total_buffer_size(l_max, GRID_DIM_X, GRID_DIM_Y, dtype, requires_grad);

    bool accepted = required_buff_size <= deviceProp.sharedMemPerBlockOptin;

    // TORCH_CHECK(
    //     required_buff_size <= deviceProp.sharedMemPerBlockOptin,
    //     "requested shared memory buffer (", required_buff_size, ") exceeds max available ",
    //     "on device: ", deviceProp.name, " (", deviceProp.sharedMemPerBlockOptin, ")"
    // );

    if (!accepted){
        printf("Warning: requested shared memory buffer (%d) exceeds max available (%d) on device (%s)\n", required_buff_size, deviceProp.sharedMemPerBlockOptin, deviceProp.name );
    } else {
        // printf("Accepted shared memory buffer (%d) max available (%d) on device (%s)\n", required_buff_size, deviceProp.sharedMemPerBlockOptin, deviceProp.name );
        switch (scalar_type) {
        case torch::ScalarType::Double:
            cudaFuncSetAttribute(
                spherical_harmonics_kernel<double>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                required_buff_size
            );
            break;
        case torch::ScalarType::Float:
            cudaFuncSetAttribute(
                spherical_harmonics_kernel<float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                required_buff_size
            );
            break;
        // case torch::ScalarType::Half:
        //     cudaFuncSetAttribute(
        //         spherical_harmonics_kernel<at::Half>,
        //         cudaFuncAttributeMaxDynamicSharedMemorySize,
        //         total_buff_size
        //     );
        //     break;
        }
    }
    return accepted;
}

std::vector<torch::Tensor> sphericart_torch::spherical_harmonics_cuda(
    torch::Tensor xyz,
    torch::Tensor prefactors,
    int64_t l_max,
    bool normalize,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool gradients
) {

    CHECK_INPUT(xyz);
    CHECK_INPUT(prefactors);
    CHECK_SAME_DTYPE(xyz, prefactors);

    int n_total = (l_max + 1) * (l_max + 1);

    auto sph = torch::empty(
        {xyz.size(0), n_total},
        torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
    );

    torch::Tensor d_sph;
    if (xyz.requires_grad() || gradients) {
        d_sph = torch::empty(
            {xyz.size(0), 3, n_total},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
        );
    } else {
        // just so accessor doesn't complain
        d_sph = torch::empty(
            {1, 1, 1},
            torch::TensorOptions().dtype(xyz.dtype()).device(xyz.device())
        );
    }

    dim3 grid_dim(GRID_DIM_X, GRID_DIM_Y);

    auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

    dim3 block_dim(find_num_blocks(xyz.size(0), GRID_DIM_X));

    int nl = max(
        static_cast<size_t>((HARDCODED_LMAX + 1) * (HARDCODED_LMAX + 1)),
         2 * l_max + 1
     );

    //int nl = 2 * l_max + 1;

    AT_DISPATCH_FLOATING_TYPES(
        xyz.scalar_type(), "spherical_harmonics_cuda", ([&] {
            size_t total_buff_size = 0;

            total_buff_size += GRID_DIM_X * (l_max + 1) * sizeof(scalar_t);     // buffer_c
            total_buff_size += GRID_DIM_X * (l_max + 1) * sizeof(scalar_t);     // buffer_s
            total_buff_size += (l_max + 1) * (l_max + 2) * sizeof(scalar_t);    // buffer_prefactors
            total_buff_size += GRID_DIM_Y * GRID_DIM_X * nl * sizeof(scalar_t); // buffer_sph_out

            if (xyz.requires_grad() || gradients) {
                total_buff_size += 3 * GRID_DIM_Y * GRID_DIM_X * nl * sizeof(scalar_t); // buffer_sph_derivs
            }

            //printf("total buff size: %d\n", total_buff_size);

            spherical_harmonics_kernel<<<block_dim, grid_dim, total_buff_size>>>(
                xyz.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                prefactors.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                l_max,
                xyz.requires_grad() || gradients,
                normalize,
                sph.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                d_sph.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    
    cudaDeviceSynchronize();

    if (xyz.requires_grad() || gradients) {
        return {sph, d_sph};
    } else {
        return {sph, torch::Tensor()};
    }
}

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void backward_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> dsph,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> sph_grad,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_grad
) {

        /*
        for (int spatial = 0; spatial < 3; spatial++) {
            auto gradient_slice = dsph.index(
                {torch::indexing::Slice(), spatial, torch::indexing::Slice()}
            );
            xyz_grad.index_put_(
                {torch::indexing::Slice(), spatial},
                torch::sum(sph_grad * gradient_slice, 1)
            );
        }*/
    
    int atom_idx = blockIdx.x * blockDim.y + threadIdx.y;
    
    int spatial = blockIdx.y;

    scalar_t sum = 0.0;

    for (int j = threadIdx.x; j < sph_grad.size(1); j +=blockDim.x){
        sum +=  dsph[atom_idx][spatial][j] * sph_grad[atom_idx][j];
    }

    // reduce across the sub-warp
    for (int offset = blockDim.x/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x == 0) {
        xyz_grad[atom_idx][spatial]  = sum;
    }

}

torch::Tensor sphericart_torch::spherical_harmonics_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor dsph,
    torch::Tensor sph_grad
) {

    if (!xyz.device().is_cuda()) {
        throw std::runtime_error("internal error: CUDA version called on non-CUDA tensor");
    }

    auto xyz_grad = torch::Tensor();

    if (xyz.requires_grad()) {
        xyz_grad = torch::empty_like(xyz);

        dim3 grid_dim(4, 32);

        auto find_num_blocks = [](int x, int bdim) { return (x + bdim - 1) / bdim; };

        dim3 block_dim(find_num_blocks(xyz.size(0), 32), 3);

        AT_DISPATCH_FLOATING_TYPES(
        xyz.scalar_type(), "spherical_harmonics_backward_cuda", ([&] {
           
            backward_kernel<<<block_dim, grid_dim>>>(
                dsph.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                sph_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                xyz_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
        }));

    cudaDeviceSynchronize();

    }

    return xyz_grad;
}

template<typename T>
void compute_sph_prefactors(int l_max, T *factors) {
    auto k = 0; // quick access index
    for (int l = 0; l <= l_max; ++l) {
        T factor = (2 * l + 1) / (2 * M_PI);
        // incorporates  the 1/sqrt(2) that goes with the m=0 SPH
        factors[k] = sqrt(factor) * M_SQRT1_2;
        for (int m = 1; m <= l; ++m) {
            factor *= 1.0 / (l * (l + 1) + m * (1 - m));
            if (m % 2 == 0) {
                factors[k + m] = sqrt(factor);
            } else {
                factors[k + m] = -sqrt(factor);
            }
        }
        k += l + 1;
    }

    // that are needed in the recursive calculation of Qlm.
    // Xll is just Qll, Xlm is the factor that enters the alternative m recursion
    factors[k] = 1.0; k += 1;
    for (int l = 1; l < l_max + 1; l++) {
        factors[k+l] = -(2 * l - 1) * factors[k - 1];
        for (int m = l - 1; m >= 0; --m) {
            factors[k + m] = -1.0 / ((l + m + 1) * (l - m));
        }
        k += l + 1;
    }
}

torch::Tensor sphericart_torch::prefactors_cuda(int64_t l_max, at::ScalarType dtype) {
    auto result = torch::empty({(l_max + 1) * (l_max + 2)}, torch::TensorOptions().device("cpu").dtype(dtype));

    if (dtype == c10::kDouble) {
        compute_sph_prefactors(l_max, static_cast<double*>(result.data_ptr()));
    } else if (dtype == c10::kFloat) {
        compute_sph_prefactors(l_max, static_cast<float*>(result.data_ptr()));
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }

    return result.to("cuda");
}
