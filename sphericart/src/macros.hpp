#ifndef SPHERICART_MACROS_HPP
#define SPHERICART_MACROS_HPP

/*
    Hard-coded expressions for low-l Cartesian spherical harmonics and their derivatives.

    These are implemented as macros to be included anywhere at compile time. They assume
    values are computed for one point at a time, and that the spherical harmonics are
    stored in a contiguous section that "flattens" the (l,m) dimensions, e.g.
    [ (0,0), (1,-1), (1,0), (1,1), (2,-2), ...]
    Functions get pointers to the beginning of the storage space for the current sample,
    x,y,z and, for l>1, x^2, y^2 and z^2, which can be reused.
    Each macro computes one l, and macros should be called in order as the higher l
    reuse calculations at lower angular momentum. The expressions here are derived
    with computer assisted algebra by attempting all possible polynomial decompositions
    and selecting that with the smallest number of operations.
*/

#define SPHERICART_LMAX_HARDCODED 6

#define SPH_IDX_IDEN 

#define COMPUTE_SPH_L0(sph_i, SPH_IDX) \
    (sph_i)[SPH_IDX(0)] = 0.282094791773878;

#define COMPUTE_SPH_DERIVATIVE_L0(sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX) \
    (dxsph_i)[SPH_IDX(0)] = (dysph_i)[SPH_IDX(0)] = (dzsph_i)[SPH_IDX(0)] = 0.0;

#define COMPUTE_SPH_L1(x, y, z, sph_i, SPH_IDX)                                                                                \
    (sph_i)[SPH_IDX(1)] = 0.48860251190292 * y;                                                                                 \
    (sph_i)[SPH_IDX(2)] = 0.48860251190292 * z;                                                                                 \
    (sph_i)[SPH_IDX(3)] = 0.48860251190292 * x;

#define COMPUTE_SPH_DERIVATIVE_L1(sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                                             \
    (dxsph_i)[SPH_IDX(1)] = 0.0;                                                                                                \
    (dxsph_i)[SPH_IDX(2)] = 0.0;                                                                                                \
    (dxsph_i)[SPH_IDX(3)] = 0.48860251190292;                                                                                   \
    (dysph_i)[SPH_IDX(1)] = 0.48860251190292;                                                                                   \
    (dysph_i)[SPH_IDX(2)] = 0.0;                                                                                                \
    (dysph_i)[SPH_IDX(3)] = 0.0;                                                                                                \
    (dzsph_i)[SPH_IDX(1)] = 0.0;                                                                                                \
    (dzsph_i)[SPH_IDX(2)] = 0.48860251190292;                                                                                   \
    (dzsph_i)[SPH_IDX(3)] = 0.0;

#define COMPUTE_SPH_L2(x, y, z, x2, y2, z2, sph_i, SPH_IDX)                                                                    \
    {                                                                                                                  \
        auto tmp = 2.23606797749979 * x;                                                                                    \
        (sph_i)[SPH_IDX(4)] = tmp * (sph_i)[SPH_IDX(1)];                                                                                 \
        (sph_i)[SPH_IDX(7)] = tmp * (sph_i)[SPH_IDX(2)];                                                                                 \
        (sph_i)[SPH_IDX(5)] = 2.23606797749979 * z * (sph_i)[SPH_IDX(1)];                                                                \
        (sph_i)[SPH_IDX(6)] = -0.315391565252520 * (x2 + y2 - 2 * z2);                                                          \
        (sph_i)[SPH_IDX(8)] = 0.54627421529604 * (x2 - y2);                                                                     \
    }

#define COMPUTE_SPH_DERIVATIVE_L2(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                        \
    (dxsph_i)[SPH_IDX(4)] = 2.23606797749979 * (sph_i)[SPH_IDX(1)];                                                                      \
    (dxsph_i)[SPH_IDX(5)] = 0.0;                                                                                                \
    (dxsph_i)[SPH_IDX(6)] = -1.29099444873581 * (sph_i)[SPH_IDX(3)];                                                                     \
    (dxsph_i)[SPH_IDX(7)] = 2.23606797749979 * (sph_i)[SPH_IDX(2)];                                                                      \
    (dxsph_i)[SPH_IDX(8)] = 2.23606797749979 * (sph_i)[SPH_IDX(3)];                                                                      \
                                                                                                                       \
    (dysph_i)[SPH_IDX(4)] = -1.73205080756888 * (dxsph_i)[SPH_IDX(6)];                                                                   \
    (dysph_i)[SPH_IDX(5)] = (dxsph_i)[SPH_IDX(7)];                                                                                       \
    (dysph_i)[SPH_IDX(6)] = -0.577350269189626 * (dxsph_i)[SPH_IDX(4)];                                                                  \
    (dysph_i)[SPH_IDX(7)] = 0.0;                                                                                                \
    (dysph_i)[SPH_IDX(8)] = -(dxsph_i)[SPH_IDX(4)];                                                                                      \
                                                                                                                       \
    (dzsph_i)[SPH_IDX(4)] = (dzsph_i)[SPH_IDX(8)] = 0.0;                                                                                 \
    (dzsph_i)[SPH_IDX(5)] = (dxsph_i)[SPH_IDX(4)];                                                                                       \
    (dzsph_i)[SPH_IDX(6)] = 1.15470053837925 * (dxsph_i)[SPH_IDX(7)];                                                                    \
    (dzsph_i)[SPH_IDX(7)] = (dysph_i)[SPH_IDX(4)];

#define COMPUTE_SPH_L3(x, y, z, x2, y2, z2, sph_i, SPH_IDX)                                                                    \
    {                                                                                                                  \
        sph_i[SPH_IDX(9)] = -0.59004358992664 * y * (y2 - 3 * x2);                                                              \
        sph_i[SPH_IDX(10)] = 2.64575131106459 * z * sph_i[SPH_IDX(4)];                                                                   \
        auto tmp = -0.457045799464466 * (x2 + y2 - 4 * z2);                                                                 \
        sph_i[SPH_IDX(11)] = y * tmp;                                                                                           \
        sph_i[SPH_IDX(13)] = x * tmp;                                                                                           \
        sph_i[SPH_IDX(12)] = -1.49270533036046 * z * (z2 - 2.37799637856361 * sph_i[SPH_IDX(6)]);                                        \
        sph_i[SPH_IDX(14)] = 1.44530572132028 * z * (x2 - y2);                                                                  \
        sph_i[SPH_IDX(15)] = 0.59004358992664 * x * (x2 - 3 * y2);                                                              \
    }

#define COMPUTE_SPH_DERIVATIVE_L3(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                        \
    dxsph_i[SPH_IDX(9)] = 3.24037034920393 * sph_i[SPH_IDX(4)];                                                                          \
    dxsph_i[SPH_IDX(10)] = 2.64575131106459 * sph_i[SPH_IDX(5)];                                                                         \
    dxsph_i[SPH_IDX(11)] = -0.83666002653408 * sph_i[SPH_IDX(4)];                                                                        \
    dxsph_i[SPH_IDX(12)] = -2.04939015319192 * sph_i[SPH_IDX(7)];                                                                        \
    dxsph_i[SPH_IDX(13)] = 0.91409159892893 * (y2 - z2 + 4.75599275712721 * sph_i[SPH_IDX(6)]);                                          \
    dxsph_i[SPH_IDX(14)] = 2.64575131106459 * sph_i[SPH_IDX(7)];                                                                         \
    dxsph_i[SPH_IDX(15)] = 3.24037034920393 * sph_i[SPH_IDX(8)];                                                                         \
                                                                                                                       \
    dysph_i[SPH_IDX(9)] = dxsph_i[SPH_IDX(15)];                                                                                          \
    dysph_i[SPH_IDX(10)] = dxsph_i[SPH_IDX(14)];                                                                                         \
    dysph_i[SPH_IDX(11)] = -0.91409159892893 * (y2 - z2 - 1.58533091904240 * sph_i[SPH_IDX(6)]);                                         \
    dysph_i[SPH_IDX(12)] = -2.04939015319192 * sph_i[SPH_IDX(5)];                                                                        \
    dysph_i[SPH_IDX(13)] = -0.83666002653408 * sph_i[SPH_IDX(4)];                                                                        \
    dysph_i[SPH_IDX(14)] = -dxsph_i[SPH_IDX(10)];                                                                                        \
    dysph_i[SPH_IDX(15)] = -dxsph_i[SPH_IDX(9)];                                                                                         \
                                                                                                                       \
    dzsph_i[SPH_IDX(9)] = 0.0;                                                                                                  \
    dzsph_i[SPH_IDX(10)] = 2.64575131106459 * sph_i[SPH_IDX(4)];                                                                         \
    dzsph_i[SPH_IDX(11)] = 3.34664010613630 * sph_i[SPH_IDX(5)];                                                                         \
    dzsph_i[SPH_IDX(12)] = 3.54964786985977 * sph_i[SPH_IDX(6)];                                                                         \
    dzsph_i[SPH_IDX(13)] = 3.34664010613630 * sph_i[SPH_IDX(7)];                                                                         \
    dzsph_i[SPH_IDX(14)] = 2.64575131106459 * sph_i[SPH_IDX(8)];                                                                         \
    dzsph_i[SPH_IDX(15)] = 0.0;

#define COMPUTE_SPH_L4(x, y, z, x2, y2, z2, sph_i, SPH_IDX)                                                                    \
    {                                                                                                                  \
        sph_i[SPH_IDX(16)] = 4.194391357527674 * sph_i[SPH_IDX(4)] * sph_i[SPH_IDX(8)];                                                           \
        sph_i[SPH_IDX(17)] = 3 * z * sph_i[SPH_IDX(9)];                                                                                  \
        auto tmp = -0.866025403784439 * (x2 + y2 - 6 * z2);                                                                 \
        sph_i[SPH_IDX(18)] = tmp * sph_i[SPH_IDX(4)];                                                                                    \
        sph_i[SPH_IDX(22)] = tmp * sph_i[SPH_IDX(8)];                                                                                    \
        sph_i[SPH_IDX(20)] = -0.69436507482941 * (y * sph_i[SPH_IDX(11)] - 1.6329931618554521 * z * sph_i[SPH_IDX(12)] + x * sph_i[SPH_IDX(13)]);          \
        tmp = -1.224744871391589 * (z2 - 4.755992757127213 * sph_i[SPH_IDX(6)]);                                                \
        sph_i[SPH_IDX(19)] = sph_i[SPH_IDX(5)] * tmp;                                                                                    \
        sph_i[SPH_IDX(21)] = sph_i[SPH_IDX(7)] * tmp;                                                                                    \
        sph_i[SPH_IDX(23)] = 3 * z * sph_i[SPH_IDX(15)];                                                                                 \
        sph_i[SPH_IDX(24)] = -1.060660171779821 * (y * sph_i[SPH_IDX(9)] - x * sph_i[SPH_IDX(15)]);                                               \
    }

#define COMPUTE_SPH_DERIVATIVE_L4(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                        \
    (dxsph_i)[SPH_IDX(16)] = 4.242640687119285 * (sph_i)[SPH_IDX(9)];                                                                    \
    (dxsph_i)[SPH_IDX(17)] = 3.674234614174767 * (sph_i)[SPH_IDX(10)];                                                                   \
    (dxsph_i)[SPH_IDX(18)] = 1.892349391515120 * y * (y2 + 4.755992757127213 * (sph_i)[SPH_IDX(6)]);                                     \
    (dxsph_i)[SPH_IDX(19)] = -1.388730149658827 * (sph_i)[SPH_IDX(10)];                                                                  \
    (dxsph_i)[SPH_IDX(20)] = -2.777460299317654 * (sph_i)[SPH_IDX(13)];                                                                  \
    (dxsph_i)[SPH_IDX(21)] =                                                                                                    \
        -1.338093087114578 * (z * z2 - 2.745873698591307 * y * (sph_i)[SPH_IDX(5)] - 4.019547514144073 * (sph_i)[SPH_IDX(12)]);          \
    (dxsph_i)[SPH_IDX(22)] = -1.892349391515120 * x * (x2 - 3 * z2);                                                            \
    (dxsph_i)[SPH_IDX(23)] = 3.674234614174767 * (sph_i)[SPH_IDX(14)];                                                                   \
    (dxsph_i)[SPH_IDX(24)] = 4.242640687119285 * (sph_i)[SPH_IDX(15)];                                                                   \
                                                                                                                       \
    (dysph_i)[SPH_IDX(16)] = (dxsph_i)[SPH_IDX(24)];                                                                                     \
    (dysph_i)[SPH_IDX(17)] = (dxsph_i)[SPH_IDX(23)];                                                                                     \
    (dysph_i)[SPH_IDX(18)] = -1.892349391515120 * x * (y2 - 2 * z2 - 1.585330919042404 * (sph_i)[SPH_IDX(6)]);                           \
    (dysph_i)[SPH_IDX(19)] = -1.338093087114578 * (z * (3 * y2 - z2) - 1.339849171381358 * (sph_i)[SPH_IDX(12)]);                        \
    (dysph_i)[SPH_IDX(20)] = -2.777460299317654 * (sph_i)[SPH_IDX(11)];                                                                  \
    (dysph_i)[SPH_IDX(21)] = (dxsph_i)[SPH_IDX(19)];                                                                                     \
    (dysph_i)[SPH_IDX(22)] = 1.892349391515120 * y * (y2 - 3 * z2);                                                             \
    (dysph_i)[SPH_IDX(23)] = -(dxsph_i)[SPH_IDX(17)];                                                                                    \
    (dysph_i)[SPH_IDX(24)] = -(dxsph_i)[SPH_IDX(16)];                                                                                    \
                                                                                                                       \
    (dzsph_i)[SPH_IDX(16)] = 0.0;                                                                                               \
    (dzsph_i)[SPH_IDX(17)] = 3 * (sph_i)[SPH_IDX(9)];                                                                                    \
    (dzsph_i)[SPH_IDX(18)] = 3.927922024247863 * (sph_i)[SPH_IDX(10)];                                                                   \
    (dzsph_i)[SPH_IDX(19)] = 4.391550328268399 * (sph_i)[SPH_IDX(11)];                                                                   \
    (dzsph_i)[SPH_IDX(20)] = 4.535573676110727 * (sph_i)[SPH_IDX(12)];                                                                   \
    (dzsph_i)[SPH_IDX(21)] = 4.391550328268399 * (sph_i)[SPH_IDX(13)];                                                                   \
    (dzsph_i)[SPH_IDX(22)] = 3.927922024247863 * (sph_i)[SPH_IDX(14)];                                                                   \
    (dzsph_i)[SPH_IDX(23)] = 3 * (sph_i)[SPH_IDX(15)];                                                                                   \
    (dzsph_i)[SPH_IDX(24)] = 0.0;

#define COMPUTE_SPH_L5(x, y, z, x2, y2, z2, sph_i, SPH_IDX)                                                                    \
    {                                                                                                                  \
        sph_i[SPH_IDX(25)] = 13.12764113680340 * y * (y2 * (x2 - 0.2 * y2) + 0.3994658435740642 * sph_i[SPH_IDX(24)]);                   \
        auto tmp = 3.316624790355400 * z;                                                                                   \
        sph_i[SPH_IDX(26)] = tmp * sph_i[SPH_IDX(16)];                                                                                   \
        sph_i[SPH_IDX(34)] = tmp * sph_i[SPH_IDX(24)];                                                                                   \
        tmp = 4.974937185533100 * (z2 + 0.5284436396808015 * sph_i[SPH_IDX(6)]);                                                \
        sph_i[SPH_IDX(27)] = tmp * sph_i[SPH_IDX(9)];                                                                                    \
        sph_i[SPH_IDX(33)] = tmp * sph_i[SPH_IDX(15)];                                                                                   \
        tmp = 5.257947827012948 * sph_i[SPH_IDX(6)];                                                                            \
        sph_i[SPH_IDX(28)] = tmp * sph_i[SPH_IDX(10)];                                                                                   \
        sph_i[SPH_IDX(32)] = tmp * sph_i[SPH_IDX(14)];                                                                                   \
        tmp = 0.6324555320336759 * z;                                                                                  \
        sph_i[SPH_IDX(29)] = 1.427248064296125 * (y * sph_i[SPH_IDX(20)] + tmp * sph_i[SPH_IDX(19)]);                                             \
        sph_i[SPH_IDX(31)] = 1.427248064296125 * (x * sph_i[SPH_IDX(20)] + tmp * sph_i[SPH_IDX(21)]);                                             \
        sph_i[SPH_IDX(30)] = 1.403403869441083 * (3.540173863740353 * sph_i[SPH_IDX(6)] * sph_i[SPH_IDX(12)] - z * z2 * z2);                      \
        sph_i[SPH_IDX(35)] = -1.048808848170152 * (y * sph_i[SPH_IDX(16)] - x * sph_i[SPH_IDX(24)]);                                              \
    }

#define COMPUTE_SPH_DERIVATIVE_L5(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                        \
    (dxsph_i)[SPH_IDX(25)] = 5.244044240850758 * (sph_i)[SPH_IDX(16)];                                                                   \
    (dxsph_i)[SPH_IDX(26)] = 4.690415759823430 * (sph_i)[SPH_IDX(17)];                                                                   \
    (dxsph_i)[SPH_IDX(27)] = 3.582364210034113 * (y2 * (sph_i)[SPH_IDX(4)] + 3.58568582800318 * x * (sph_i)[SPH_IDX(11)]);                        \
    (dxsph_i)[SPH_IDX(28)] = -8.774964387392122 * ((y2 - z2) * (sph_i)[SPH_IDX(5)] + 0.3086066999241838 * (sph_i)[SPH_IDX(17)]);                  \
    (dxsph_i)[SPH_IDX(29)] = -1.914854215512676 * (sph_i)[SPH_IDX(18)];                                                                  \
    (dxsph_i)[SPH_IDX(30)] = -3.496029493900505 * (sph_i)[SPH_IDX(21)];                                                                  \
    (dxsph_i)[SPH_IDX(31)] = -8.616843969807043 * (0.2102610435016800 * z2 * z2 + 1.056887279361603 * (sph_i)[SPH_IDX(5)] * (sph_i)[SPH_IDX(5)] + \
                                          (y2 - z2) * (sph_i)[SPH_IDX(6)] + 0.555555555555556 * (sph_i)[SPH_IDX(22)]);                   \
    (dxsph_i)[SPH_IDX(32)] = -8.774964387392122 * (x2 - z2) * (sph_i)[SPH_IDX(7)];                                                       \
    (dxsph_i)[SPH_IDX(33)] = -5.170697352496190 *                                                                               \
                    (0.106904496764970 * z * (dxsph_i)[SPH_IDX(23)] - 0.320713490294909 * y * (sph_i)[SPH_IDX(9)] - (sph_i)[SPH_IDX(22)]);        \
    (dxsph_i)[SPH_IDX(34)] = 4.690415759823430 * (sph_i)[SPH_IDX(23)];                                                                   \
    (dxsph_i)[SPH_IDX(35)] = 5.24404424085076 * (sph_i)[SPH_IDX(24)];                                                                    \
                                                                                                                       \
    (dysph_i)[SPH_IDX(25)] = (dxsph_i)[SPH_IDX(35)];                                                                                     \
    (dysph_i)[SPH_IDX(26)] = (dxsph_i)[SPH_IDX(34)];                                                                                     \
    (dysph_i)[SPH_IDX(27)] =                                                                                                    \
        -3.102418411497714 * (0.534522483824849 * y * (sph_i)[SPH_IDX(9)] - 0.654653670707977 * z * (sph_i)[SPH_IDX(14)] - (sph_i)[SPH_IDX(22)]); \
    (dysph_i)[SPH_IDX(28)] = -8.77496438739212 * (y2 - 1.585330919042404 * (sph_i)[SPH_IDX(6)]) * (sph_i)[SPH_IDX(7)];                            \
    (dysph_i)[SPH_IDX(29)] = 0.7237468644557459 * (y * (2.12132034355964 * (sph_i)[SPH_IDX(9)] - 8.21583836257749 * (sph_i)[SPH_IDX(11)]) +       \
                                          6.70820393249937 * z * (sph_i)[SPH_IDX(12)] + (sph_i)[SPH_IDX(24)]);                           \
    (dysph_i)[SPH_IDX(30)] = -3.496029493900505 * (sph_i)[SPH_IDX(19)];                                                                  \
    (dysph_i)[SPH_IDX(31)] = (dxsph_i)[SPH_IDX(29)];                                                                                     \
    (dysph_i)[SPH_IDX(32)] = 8.77496438739212 * (y2 - z2) * (sph_i)[SPH_IDX(5)];                                                         \
    (dysph_i)[SPH_IDX(33)] = 3.582364210034113 * (sph_i)[SPH_IDX(4)] * (y2 - 5 * z2 - 1.585330919042404 * (sph_i)[SPH_IDX(6)]);                   \
    (dysph_i)[SPH_IDX(34)] = -(dxsph_i)[SPH_IDX(26)];                                                                                    \
    (dysph_i)[SPH_IDX(35)] = -(dxsph_i)[SPH_IDX(25)];                                                                                    \
                                                                                                                       \
    (dzsph_i)[SPH_IDX(25)] = 0.0;                                                                                               \
    (dzsph_i)[SPH_IDX(26)] = 3.316624790355400 * (sph_i)[SPH_IDX(16)];                                                                   \
    (dzsph_i)[SPH_IDX(27)] = 4.422166387140533 * (sph_i)[SPH_IDX(17)];                                                                   \
    (dzsph_i)[SPH_IDX(28)] = 5.066228051190221 * (sph_i)[SPH_IDX(18)];                                                                   \
    (dzsph_i)[SPH_IDX(29)] = 5.416025603090640 * (sph_i)[SPH_IDX(19)];                                                                   \
    (dzsph_i)[SPH_IDX(30)] = 5.527707983925666 * (sph_i)[SPH_IDX(20)];                                                                   \
    (dzsph_i)[SPH_IDX(31)] = 5.416025603090640 * (sph_i)[SPH_IDX(21)];                                                                   \
    (dzsph_i)[SPH_IDX(32)] = 5.066228051190221 * (sph_i)[SPH_IDX(22)];                                                                   \
    (dzsph_i)[SPH_IDX(33)] = 4.422166387140533 * (sph_i)[SPH_IDX(23)];                                                                   \
    (dzsph_i)[SPH_IDX(34)] = 3.316624790355400 * (sph_i)[SPH_IDX(24)];                                                                   \
    (dzsph_i)[SPH_IDX(35)] = 0.0;

#define COMPUTE_SPH_L6(x, y, z, x2, y2, z2, sph_i, SPH_IDX)                                                                    \
    {                                                                                                                  \
        (sph_i)[SPH_IDX(36)] = 3.924637560539857 * (sph_i)[SPH_IDX(9)] * (sph_i)[SPH_IDX(15)];                                                    \
        auto tmp = 3.605551275463989 * z;                                                                                   \
        (sph_i)[SPH_IDX(37)] = tmp * (sph_i)[SPH_IDX(25)];                                                                               \
        (sph_i)[SPH_IDX(47)] = tmp * (sph_i)[SPH_IDX(35)];                                                                               \
        tmp = 6.4498061986388 * (z2 + 0.396332729760601 * (sph_i)[SPH_IDX(6)]);                                                 \
        (sph_i)[SPH_IDX(38)] = tmp * (sph_i)[SPH_IDX(16)];                                                                               \
        (sph_i)[SPH_IDX(46)] = tmp * (sph_i)[SPH_IDX(24)];                                                                               \
        tmp = 1.04083299973307 * (z2 + 4.75599275712721 * (sph_i)[SPH_IDX(6)]);                                                 \
        (sph_i)[SPH_IDX(39)] = tmp * (sph_i)[SPH_IDX(17)];                                                                               \
        (sph_i)[SPH_IDX(45)] = tmp * sph_i[SPH_IDX(23)];                                                                                 \
        (sph_i)[SPH_IDX(40)] = 2.033805211017918 * (0.3779644730092272 * z * (sph_i)[SPH_IDX(28)] + x * (sph_i)[SPH_IDX(29)]);                    \
        tmp = -6.399218702310463 * (z2 * z2 - 4.188790204786391 * (sph_i)[SPH_IDX(6)] * (sph_i)[SPH_IDX(6)]);                            \
        (sph_i)[SPH_IDX(41)] = tmp * (sph_i)[SPH_IDX(5)];                                                                                \
        (sph_i)[SPH_IDX(43)] = tmp * (sph_i)[SPH_IDX(7)];                                                                                \
        (sph_i)[SPH_IDX(42)] = -1.087114613009218 *                                                                             \
                      (0.645497224367903 * y * (sph_i)[SPH_IDX(29)] - z * (sph_i)[SPH_IDX(30)] + 0.645497224367903 * x * (sph_i)[SPH_IDX(31)]);   \
        (sph_i)[SPH_IDX(44)] = -0.9414688716912718 * (y * (sph_i)[SPH_IDX(27)] - 1.63299316185545 * z * (sph_i)[SPH_IDX(32)] + x * (sph_i)[SPH_IDX(33)]);  \
        (sph_i)[SPH_IDX(48)] = -1.040832999733066 * (y * (sph_i)[SPH_IDX(25)] - x * (sph_i)[SPH_IDX(35)]);                                        \
    }

#define COMPUTE_SPH_DERIVATIVE_L6(x, y, z, x2, y2, z2, sph_i, dxsph_i, dysph_i, dzsph_i, SPH_IDX)                                        \
    {                                                                                                                  \
        (dxsph_i)[SPH_IDX(36)] = 6.244997998398398 * (sph_i)[SPH_IDX(25)];                                                               \
        (dysph_i)[SPH_IDX(48)] = -(dxsph_i)[SPH_IDX(36)];                                                                                \
        (dxsph_i)[SPH_IDX(37)] = 5.700877125495690 * (sph_i)[SPH_IDX(26)];                                                               \
        (dysph_i)[SPH_IDX(47)] = -(dxsph_i)[SPH_IDX(37)];                                                                                \
        (dxsph_i)[SPH_IDX(38)] = -8.07303841165959 * y *                                                                        \
                        (y2 * y2 - 4.188790204786391 * (sph_i)[SPH_IDX(5)] * (sph_i)[SPH_IDX(5)] - 2.642218198404007 * (sph_i)[SPH_IDX(22)]);     \
        (dxsph_i)[SPH_IDX(39)] = -15.29705854077835 * ((y2 - z2) * (sph_i)[SPH_IDX(10)] + 0.2611164839335468 * (sph_i)[SPH_IDX(26)]);             \
        (dxsph_i)[SPH_IDX(40)] =                                                                                                \
            32.08092506951781 * ((sph_i)[SPH_IDX(5)] * (0.577350269189626 * y * (sph_i)[SPH_IDX(5)] - z * (sph_i)[SPH_IDX(6)]) +                  \
                                 0.364182810197360 * y * y2 * (sph_i)[SPH_IDX(6)] + 0.3169804496925759 * (sph_i)[SPH_IDX(29)]);          \
        (dxsph_i)[SPH_IDX(41)] = -2.430862174021989 * (sph_i)[SPH_IDX(28)];                                                              \
        (dysph_i)[SPH_IDX(43)] = (dxsph_i)[SPH_IDX(41)];                                                                                 \
        (dxsph_i)[SPH_IDX(42)] = -4.210376791603422 * (sph_i)[SPH_IDX(31)];                                                              \
        (dysph_i)[SPH_IDX(42)] = -4.210376791603422 * (sph_i)[SPH_IDX(29)];                                                              \
        (dxsph_i)[SPH_IDX(43)] = 4.660970900149851 * (z2 * z * (1.666666666666667 * y2 + z2 - 2.642218198404007 * (sph_i)[SPH_IDX(6)]) + \
                                             1.245553603643984 * y * (sph_i)[SPH_IDX(19)] + 1.781383145961857 * (sph_i)[SPH_IDX(30)]);   \
        (dxsph_i)[SPH_IDX(44)] = 14.73928415223878 * (x * (y2 - z2) * (2 * x2 - z2 - y2) + 0.2856568031469765 * (sph_i)[SPH_IDX(35)]);   \
        (dxsph_i)[SPH_IDX(45)] = 3.122498999199199 *                                                                            \
                        (y * (sph_i)[SPH_IDX(17)] - 1.224744871391589 * z2 * (sph_i)[SPH_IDX(14)] + +1.846372364689991 * (sph_i)[SPH_IDX(32)]);   \
        auto tmp = 1.612451549659710 * (y * (sph_i)[SPH_IDX(16)] - 1.4142135623730950 * z * (sph_i)[SPH_IDX(23)]);                            \
        (dxsph_i)[SPH_IDX(46)] = tmp + 6.18796485857095 * (sph_i)[SPH_IDX(33)];                                                          \
        (dysph_i)[SPH_IDX(38)] = -tmp + 4.125309905713972 * (sph_i)[SPH_IDX(33)];                                                        \
        (dxsph_i)[SPH_IDX(47)] = 5.700877125495690 * (sph_i)[SPH_IDX(34)];                                                               \
        (dxsph_i)[SPH_IDX(48)] = 6.244997998398398 * (sph_i)[SPH_IDX(35)];                                                               \
        (dysph_i)[SPH_IDX(36)] = (dxsph_i)[SPH_IDX(48)];                                                                                 \
        (dysph_i)[SPH_IDX(37)] = (dxsph_i)[SPH_IDX(47)];                                                                                 \
        (dysph_i)[SPH_IDX(39)] = -3.122498999199199 *                                                                           \
                        (-1.22474487139159 * z2 * (sph_i)[SPH_IDX(14)] + y * (sph_i)[SPH_IDX(17)] - 1.10782341881399 * (sph_i)[SPH_IDX(32)]);     \
        (dysph_i)[SPH_IDX(40)] =                                                                                                \
            11.68332144554792 * (x * (-1.585330919042404 * (sph_i)[SPH_IDX(5)] * (sph_i)[SPH_IDX(5)] + (z2 - y2) * (sph_i)[SPH_IDX(6)]) +         \
                                 0.1740776559556978 * sph_i[SPH_IDX(31)]);                                                      \
        (dysph_i)[SPH_IDX(41)] =                                                                                                \
            -6.99145635022478 * z * (z2 * z2 + (5.28443639680801 * y2 - 4.188790204786391 * (sph_i)[SPH_IDX(6)]) * (sph_i)[SPH_IDX(6)]); \
        (dysph_i)[SPH_IDX(44)] = 13.49073756323204 *                                                                            \
                        (y2 * z * (sph_i)[SPH_IDX(5)] +                                                                         \
                         (-0.14940357616680 * x2 + 0.44821072850040 * y2 - 0.59761430466720 * z2) * (sph_i)[SPH_IDX(11)]);      \
        (dysph_i)[SPH_IDX(45)] = 7.648529270389177 * (y2 - z2 - 1.58533091904240 * (sph_i)[SPH_IDX(6)]) * (sph_i)[SPH_IDX(10)];                   \
        (dysph_i)[SPH_IDX(46)] = 11.40175425099138 * (0.2360174359706574 * y2 * y2 * y + (y2 - 3 * z2) * (sph_i)[SPH_IDX(9)] +           \
                                             0.1348399724926484 * (sph_i)[SPH_IDX(25)]);                                        \
        (dzsph_i)[SPH_IDX(36)] = 0.0;                                                                                           \
        (dzsph_i)[SPH_IDX(37)] = 3.605551275463989 * (sph_i)[SPH_IDX(25)];                                                               \
        (dzsph_i)[SPH_IDX(38)] = 4.861724348043977 * (sph_i)[SPH_IDX(26)];                                                               \
        (dzsph_i)[SPH_IDX(39)] = 5.64881323014763 * (sph_i)[SPH_IDX(27)];                                                                \
        (dzsph_i)[SPH_IDX(40)] = 6.14964891828646 * (sph_i)[SPH_IDX(28)];                                                                \
        (dzsph_i)[SPH_IDX(41)] = 6.43145678393600 * (sph_i)[SPH_IDX(29)];                                                                \
        (dzsph_i)[SPH_IDX(42)] = 6.52268767805531 * (sph_i)[SPH_IDX(30)];                                                                \
        (dzsph_i)[SPH_IDX(43)] = 6.43145678393600 * (sph_i)[SPH_IDX(31)];                                                                \
        (dzsph_i)[SPH_IDX(44)] = 6.14964891828646 * (sph_i)[SPH_IDX(32)];                                                                \
        (dzsph_i)[SPH_IDX(45)] = 5.64881323014763 * (sph_i)[SPH_IDX(33)];                                                                \
        (dzsph_i)[SPH_IDX(46)] = 4.861724348043977 * (sph_i)[SPH_IDX(34)];                                                               \
        (dzsph_i)[SPH_IDX(47)] = 3.605551275463989 * (sph_i)[SPH_IDX(35)];                                                               \
        (dzsph_i)[SPH_IDX(48)] = 0.0;                                                                                           \
    }

#endif
