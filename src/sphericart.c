#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void compute_sph_prefactors(unsigned int l_max, double *factors) {
    /*
        Computes the prefactors for the spherical harmonics
        (-1)^|m| sqrt((2l+1)/(2pi) (l-|m|)!/(l+|m}\|)!)
        Use an iterative formula to avoid computing a ratio
        of factorials, and incorporates the 1/sqrt(2) that 
        is associated with the m=0 SPH
    */

    unsigned int k=0; // quick access index
    for (unsigned int l=0; l<=l_max; ++l) {
        double factor = (2*l+1)/(2*M_PI);
        // incorporates  the 1/sqrt(2) that goes with the m=0 SPH
        factors[k] = sqrt(factor)*M_SQRT1_2;          
        for (int m=1; m<=l; ++m) {
            factor *= 1.0/(l*(l+1)+m*(1-m));
            if (m % 2 == 0) {
                factors[k+m] = sqrt(factor);    
            } else {
                factors[k+m] = - sqrt(factor); 
            }     
        }
        k += l+1;
    }
}

inline void _compute_sph_l0(double *sph_i) {
    sph_i[0] = 0.282094791773878;
}

inline void _compute_dsph_l0(double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {    
    dxsph_i[0] = dysph_i[0] = dzsph_i[0] = 0.0;

}

void cartesian_spherical_harmonics_l0(unsigned int n_samples, double *xyz, double *sph, double *dsph) {
    #pragma omp parallel
    {
        if (dsph == NULL) {
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                _compute_sph_l0(sph+i_sample);
            }            
        } else {
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                _compute_dsph_l0(sph+i_sample, dsph+3*i_sample, dsph+3*i_sample+1, dsph+3*i_sample+2);
            }
        }
    }
}

inline void _compute_sph_l1(double x, double y, double z, double *sph_i) {
    sph_i[1] = 0.48860251190292*y;
    sph_i[2] = 0.48860251190292*z;
    sph_i[3] = 0.48860251190292*x;            
}

inline void _compute_dsph_l1(double x, double y, double z, double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {    
    dxsph_i[1] = 0.0; dxsph_i[2] = 0.0; dxsph_i[3] = 0.48860251190292;  //d/dx 
    dysph_i[1] = 0.48860251190292; dysph_i[2] = 0.0; dysph_i[3] = 0.0;  //d/dy
    dzsph_i[1] = 0.0; dzsph_i[2] = 0.48860251190292; dzsph_i[3] = 0.0;  //d/dz
}

void cartesian_spherical_harmonics_l1(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {    
    #pragma omp parallel
    {                
        if (dsph == NULL) {
            double *xyz_i, *sph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*4;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
            }            
        } else {
            double *xyz_i, *sph_i, *dsph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {                
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*4; 
                
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);

                dsph_i = dsph+i_sample*4*3;            
                _compute_dsph_l0(sph_i, dsph_i, dsph_i+4, dsph_i+4*2);
                _compute_dsph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i, 
                                dsph_i, dsph_i+4, dsph_i+4*2);
            }
        }
    }
}

inline void _compute_sph_l2(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
    double tmp;
    tmp = 2.23606797749979*x;
    sph_i[4] = tmp*sph_i[1];
    sph_i[7] = tmp*sph_i[2];
    sph_i[5] = 2.23606797749979*z*sph_i[1];
    sph_i[6] = -0.315391565252520*(x2+y2-2*z2);
    sph_i[8] = 0.54627421529604*(x2-y2);         
}

inline void _compute_dsph_l2(double x, double y, double z, double x2, double y2, double z2, double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {    
    dxsph_i[4] = 2.23606797749979*sph_i[1]; 
    dxsph_i[5] = 0.0; 
    dxsph_i[6] = -1.29099444873581*sph_i[3]; 
    dxsph_i[7] = 2.23606797749979*sph_i[2];
    dxsph_i[8] = 2.23606797749979*sph_i[3];

    dysph_i[4] = -1.73205080756888*dxsph_i[6];
    dysph_i[5] = dxsph_i[7];
    dysph_i[6] = -0.577350269189626*dxsph_i[4];
    dysph_i[7] = 0.0;
    dysph_i[8] = -dxsph_i[4];

    dzsph_i[4] = dzsph_i[8] = 0.0;
    dzsph_i[5] = dxsph_i[4];
    dzsph_i[6] = 1.15470053837925*dxsph_i[7];
    dzsph_i[7] = dysph_i[4];
}

void cartesian_spherical_harmonics_l2(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {    
    #pragma omp parallel
    {   
        double x2, y2, z2;
        if (dsph == NULL) {
            double *xyz_i, *sph_i;            
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*9;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);               
            }            
        } else {
            double *xyz_i, *sph_i, *dsph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {                
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*9;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);               
                
                dsph_i = dsph+i_sample*9*3;            
                _compute_dsph_l0(sph_i, dsph_i, dsph_i+9, dsph_i+9*2);
                _compute_dsph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i, 
                                dsph_i, dsph_i+9, dsph_i+9*2);
                _compute_dsph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+9, dsph_i+9*2);
            }
        }
    }
}

inline void _compute_sph_l3(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
    double tmp;
    sph_i[9]  = -0.59004358992664*y*(y2-3*x2);  //(3,-3)
    sph_i[10] = 2.64575131106459*z*sph_i[4];  //(3,-2)            
    tmp = -0.457045799464466*(x2+y2-4*z2);
    sph_i[11] = y*tmp; //(3,-1)
    sph_i[13] = x*tmp; //(3,1)
    sph_i[12] = -1.49270533036046*z*(z2-2.37799637856361*sph_i[6]); //(3,0)            
    sph_i[14] = 1.44530572132028*z*(x2-y2); //(3,2)
    sph_i[15]  = 0.59004358992664*x*(x2-3*y2);  //(3,-3)       
}

inline void _compute_dsph_l3(double x, double y, double z, double x2, double y2, double z2, double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {  
    dxsph_i[9]  = 3.24037034920393*sph_i[4];
    dxsph_i[10] = 2.64575131106459*sph_i[5];
    dxsph_i[11] = -0.83666002653408*sph_i[4];
    dxsph_i[12] = -2.04939015319192*sph_i[7];
    dxsph_i[13] = 0.91409159892893*(y2-z2+4.75599275712721*sph_i[6]);
    dxsph_i[14] = 2.64575131106459*sph_i[7];
    dxsph_i[15] = 3.24037034920393*sph_i[8];
    
    dysph_i[9]  = dxsph_i[15];
    dysph_i[10] = dxsph_i[14];
    dysph_i[11] = -0.91409159892893*(y2-z2-1.58533091904240*sph_i[6]); 
    dysph_i[12] = -2.04939015319192*sph_i[5];
    dysph_i[13] = -0.83666002653408*sph_i[4];
    dysph_i[14] = -dxsph_i[10];
    dysph_i[15] = -dxsph_i[9];

    dzsph_i[9] = 0.0;
    dzsph_i[10] = 2.64575131106459*sph_i[4];
    dzsph_i[11] = 3.34664010613630*sph_i[5];
    dzsph_i[12] = 3.54964786985977*sph_i[6];
    dzsph_i[13] = 3.34664010613630*sph_i[7];
    dzsph_i[14] = 2.64575131106459*sph_i[8];
    dzsph_i[15] = 0.0; 
}

void cartesian_spherical_harmonics_l3(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {    
    #pragma omp parallel
    {   
        double x2, y2, z2;
        if (dsph == NULL) {
            double *xyz_i, *sph_i;            
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*16;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);               
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i); 
            }            
        } else {
            double *xyz_i, *sph_i, *dsph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {                
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*16;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);  
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);              
                
                dsph_i = dsph+i_sample*16*3;            
                _compute_dsph_l0(sph_i, dsph_i, dsph_i+16, dsph_i+16*2);
                _compute_dsph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i, 
                                dsph_i, dsph_i+16, dsph_i+16*2);
                _compute_dsph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+16, dsph_i+16*2);
                _compute_dsph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+16, dsph_i+16*2);
            }
        }
    }
}

inline void _compute_sph_l4(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
    double tmp;
    sph_i[16] = 4.194391357527674*sph_i[4]*sph_i[8];
    sph_i[17] = 3*z*sph_i[9];
    tmp = -0.866025403784439*(x2 + y2 - 6*z2);
    sph_i[18] = tmp*sph_i[4];
    sph_i[22] = tmp*sph_i[8];
    //tgt_i[18] -> 0.2927700218845600 (Sqrt[5] z sph_i[10] + 5 Sqrt[2] y sph_i[13])
    //tgt_i[18] -> 0.1106566670344976 x (7 Sqrt[5] z sph_i[5] + 5 Sqrt[14] sph_i[11])
    //tgt_i[18] -> 0.692820323027551 sph_i[4] (5 z2 + Sqrt[5 \[Pi]] sph_i[6])
    sph_i[20] = -0.69436507482941*(y*sph_i[11] - 1.6329931618554521*z*sph_i[12] + x * sph_i[13]);
    tmp = -1.224744871391589*(z2 - 4.755992757127213*sph_i[6]);
    sph_i[19] = sph_i[5]*tmp; 
    sph_i[21] = sph_i[7]*tmp; 
    sph_i[23] = 3*z*sph_i[15];
    sph_i[24] = -1.060660171779821 * (y*sph_i[9] - x*sph_i[15]);
}

inline void _compute_dsph_l4(double x, double y, double z, double x2, double y2, double z2, double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {  
    dxsph_i[16] = 4.242640687119285 * sph_i[9];
    dxsph_i[17] = 3.674234614174767 * sph_i[10];
    dxsph_i[18] = 1.892349391515120  * y * (y2 + 4.755992757127213 * sph_i[6]);
    dxsph_i[19] = -1.388730149658827 * sph_i[10];
    dxsph_i[20] = -2.777460299317654 * sph_i[13];
    dxsph_i[21] = -1.338093087114578 * ( z *z2 -2.745873698591307* y *sph_i[5] -4.019547514144073* sph_i[12]);
    dxsph_i[22] = -1.892349391515120 * x * (x2 - 3 * z2);
    dxsph_i[23] = 3.674234614174767 * sph_i[14];
    dxsph_i[24] = 4.242640687119285 * sph_i[15];

    dysph_i[16] = dxsph_i[24];
    dysph_i[17] = dxsph_i[23];
    dysph_i[18] = -1.892349391515120*x*(y2 - 2*z2 - 1.585330919042404*sph_i[6]);
    dysph_i[19] = -1.338093087114578 * (z*(3*y2 - z2) - 1.339849171381358*sph_i[12]);
    dysph_i[20] = -2.777460299317654*sph_i[11];
    dysph_i[21] = dxsph_i[19];
    dysph_i[22] = 1.892349391515120 *y*(y2 - 3*z2);
    dysph_i[23] = -dxsph_i[17];
    dysph_i[24] = -dxsph_i[16];
    

    dzsph_i[16] = 0.0;
    dzsph_i[17] = 3 * sph_i[9];
    dzsph_i[18] = 3.927922024247863 * sph_i[10];
    dzsph_i[19] = 4.391550328268399 * sph_i[11];
    dzsph_i[20] = 4.535573676110727 * sph_i[12];
    dzsph_i[21] = 4.391550328268399 * sph_i[13];
    dzsph_i[22] = 3.927922024247863 * sph_i[14];
    dzsph_i[23] = 3*sph_i[15];
    dzsph_i[24] = 0.0;
}

void cartesian_spherical_harmonics_l4(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {    
    #pragma omp parallel
    {   
        double x2, y2, z2;
        if (dsph == NULL) {
            double *xyz_i, *sph_i;            
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*25;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);               
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i); 
                _compute_sph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);
            }            
        } else {
            double *xyz_i, *sph_i, *dsph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {                
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*25;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);  
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);  
                _compute_sph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);            
                
                dsph_i = dsph+i_sample*25*3;            
                _compute_dsph_l0(sph_i, dsph_i, dsph_i+25, dsph_i+25*2);
                _compute_dsph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i, 
                                dsph_i, dsph_i+25, dsph_i+25*2);
                _compute_dsph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+25, dsph_i+25*2);
                _compute_dsph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+25, dsph_i+25*2);
                _compute_dsph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+25, dsph_i+25*2);
            }
        }
    }
}

inline void _compute_sph_l5(double x, double y, double z, double x2, double y2, double z2, double *sph_i) {
    double tmp;
    sph_i[25] = 13.12764113680340 *y*(y2*(x2-0.2*y2)+0.3994658435740642*sph_i[24]);
    tmp = 3.316624790355400*z;
    sph_i[26] = tmp*sph_i[16];
    sph_i[34] = tmp*sph_i[24];
    tmp = 4.974937185533100 * (z2 + 0.5284436396808015*sph_i[6]);
    sph_i[27] = tmp*sph_i[9];
    sph_i[33] = tmp*sph_i[15];
    tmp = 5.257947827012948 * sph_i[6];
    sph_i[28] = tmp * sph_i[10];
    sph_i[32] = tmp * sph_i[14];
    tmp = 0.6324555320336759 *z;
    sph_i[29] = 1.427248064296125 * (tmp * sph_i[19] + y * sph_i[20]);
    sph_i[31] = 1.427248064296125 * (x *sph_i[20] + tmp*sph_i[21]);
    sph_i[30] = 1.403403869441083 * (3.540173863740353 * sph_i[6] *sph_i[12]-z*z2*z2);
    sph_i[35] = -1.048808848170152 * (y*sph_i[16] - x*sph_i[24]);
}

inline void _compute_dsph_l5(double x, double y, double z, double x2, double y2, double z2, double *sph_i, double *dxsph_i, 
                double *dysph_i, double *dzsph_i) {  
    dxsph_i[25] = 5.244044240850758 * sph_i[16];
    dxsph_i[26] = 4.690415759823430 * sph_i[17];
    dxsph_i[27] = 3.582364210034113* (y2*sph_i[4] + 3.58568582800318*x*sph_i[11]);
    dxsph_i[28] = -8.774964387392122 *(y2* sph_i[5] - z2* sph_i[5] + 0.3086066999241838*sph_i[17]);
    dxsph_i[29] = -1.914854215512676 * sph_i[18];
    dxsph_i[30] = -3.496029493900505 * sph_i[21];
    dxsph_i[31] = -8.616843969807043 * (0.2102610435016800 *z2 *z2 + 
        1.056887279361603 * sph_i[5]*sph_i[5] + (y2-z2)*sph_i[6] + 0.555555555555556 *sph_i[22]);
    dxsph_i[32] = -8.774964387392122 * (x2 - z2) * sph_i[7];
    dxsph_i[33] = -5.170697352496190 * (0.106904496764970*z*dxsph_i[23] -  
            0.320713490294909*y*sph_i[9] - sph_i[22]);
    dxsph_i[34] = 4.690415759823430 * sph_i[23];
    dxsph_i[35] = 5.24404424085076 * sph_i[24];

    dysph_i[25] = dxsph_i[35];
    dysph_i[26] = dxsph_i[34];
    dysph_i[27] = -3.102418411497714*(0.534522483824849*y*sph_i[9]
            -0.654653670707977*z*sph_i[14] - sph_i[22]);
    dysph_i[28] = -8.77496438739212 * (y2 - 1.585330919042404*sph_i[6])*sph_i[7];
    dysph_i[29] = 0.7237468644557459 * (y * (2.12132034355964 * sph_i[9] 
         -8.21583836257749 * sph_i[11]) + 6.70820393249937 * z * sph_i[12] + sph_i[24]);
    dysph_i[30] = -3.496029493900505 * sph_i[19];
    dysph_i[31] = dxsph_i[29];
    dysph_i[32] = 8.77496438739212 * (y2 - z2) *sph_i[5];
    dysph_i[33] = 3.582364210034113 * sph_i[4] *(y2 - 5*z2 - 1.585330919042404 * sph_i[6]);
    dysph_i[34] = -dxsph_i[26];
    dysph_i[35] = -dxsph_i[25];

    dzsph_i[25] = 0.0;
    dzsph_i[26] = 3.316624790355400 * sph_i[16];
    dzsph_i[27] = 4.422166387140533 * sph_i[17];
    dzsph_i[28] = 5.066228051190221 * sph_i[18];
    dzsph_i[29] = 5.416025603090640 * sph_i[19];
    dzsph_i[30] = 5.527707983925666 * sph_i[20];
    dzsph_i[31] = 5.416025603090640 * sph_i[21];
    dzsph_i[32] = 5.066228051190221 * sph_i[22];
    dzsph_i[33] = 4.422166387140533 * sph_i[23];
    dzsph_i[34] = 3.316624790355400 * sph_i[24];
    dzsph_i[35] = 0.0;
   
}

void cartesian_spherical_harmonics_l5(unsigned int n_samples, double *xyz, 
                    double *sph, double *dsph) {    
    #pragma omp parallel
    {   
        double x2, y2, z2;
        if (dsph == NULL) {
            double *xyz_i, *sph_i;            
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*25;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);               
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i); 
                _compute_sph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);
                _compute_sph_l5(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);
            }            
        } else {
            double *xyz_i, *sph_i, *dsph_i;
            #pragma omp for
            for (int i_sample=0; i_sample<n_samples; i_sample++) {                
                xyz_i = xyz+i_sample*3;
                sph_i = sph+i_sample*36;            
                _compute_sph_l0(sph_i);
                _compute_sph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i);
                x2 = xyz_i[0]*xyz_i[0]; y2 = xyz_i[1]*xyz_i[1]; z2 = xyz_i[2]*xyz_i[2]; 
                _compute_sph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);  
                _compute_sph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);  
                _compute_sph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);            
                _compute_sph_l5(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i);
                
                dsph_i = dsph+i_sample*36*3;            
                _compute_dsph_l0(sph_i, dsph_i, dsph_i+36, dsph_i+36*2);
                _compute_dsph_l1(xyz_i[0], xyz_i[1], xyz_i[2], sph_i, 
                                dsph_i, dsph_i+36, dsph_i+36*2);
                _compute_dsph_l2(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+36, dsph_i+36*2);
                _compute_dsph_l3(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+36, dsph_i+36*2);
                _compute_dsph_l4(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+36, dsph_i+36*2);
                _compute_dsph_l5(xyz_i[0], xyz_i[1], xyz_i[2], x2, y2, z2, sph_i, 
                                dsph_i, dsph_i+36, dsph_i+36*2);
            }
        }
    }
}

void cartesian_spherical_harmonics_generic(unsigned int n_samples, unsigned int l_max, 
            const double* prefactors, double *xyz, double *sph, double *dsph) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and 
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the 
        r^l term can be easily incorporated into any radial function or 
        added a posteriori (with the corresponding derivative).
    */

    #pragma omp parallel
    {
        // storage arrays for Qlm (modified associated Legendre polynomials)
        // and terms corresponding to (scaled) cosine and sine of the azimuth
        double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        double* c = (double*) malloc(sizeof(double)*(l_max+1));
        double* s = (double*) malloc(sizeof(double)*(l_max+1));

        // temporaries to store prefactor*q and dq
        double pq, pdq, pdqx, pdqy; 
        int size_y = (l_max+1)*(l_max+1);

        /* k is a utility index to traverse lm arrays. we store sph in 
           a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
           so we often write a nested loop on l and m and track where we
           got by incrementing a separate index k. */
        int k; 
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {

            double x = xyz[i_sample*3+0];
            double y = xyz[i_sample*3+1];
            double z = xyz[i_sample*3+2];
            double r_sq = x*x+y*y+z*z;

            // pointer to the segment that should store the i_sample sph
            double *sph_i = sph+i_sample*size_y;
            
            /* compute recursively the "Cartesian" associated Legendre polynomials Qlm.
               Qlm is defined as r^l/r_xy^m P_lm, and is a polynomial of x,y,z.
               These are computed with a recursive expression.
              */
            
            // Initialize the recursion
            q[0+0] = 1.0;
            k=1;
            for (int m = 1; m < l_max+1; m++) {
                q[k+m] = -(2*m-1)*q[k-1];
                q[k+(m-1)] = -z*q[k+m]; // (2*m-1)*z*q[k-1];
                k += m+1; 
            }

            // base index to traverse the Qlm. the initial index for q[lm] starts at l=2
            k = 3; 
            for (int l=2; l < l_max+1; ++l) {
                double twolz = (2*l-1)*z;
                double lmrsq = (l-2)*r_sq;
                for (int m=0; m < l-1; ++m) {
                    lmrsq += r_sq; // this computes (l-1+m) r_sq
                    q[k] = (twolz*q[k-l]-lmrsq*q[k-(2*l-1)])/(l-m);
                    ++k; 
                }
                k += 2; // we must skip the 2 that are already precomputed
            }

            /* These are scaled version of cos(m phi) and sin(m phi).
               Basically, these are cos and sin multiplied by r_xy^m,
               so that they are just plain polynomials of x,y,z.
            */
            c[0] = 1.0;
            s[0] = 0.0;
            for (int m = 1; m < l_max+1; m++) {
                c[m] = c[m-1]*x-s[m-1]*y;
                s[m] = c[m-1]*y+s[m-1]*x;
            }

            /* fill the (Cartesian) sph by combining Qlm and 
              sine/cosine phi-dependent factors. we use pointer 
              arithmetics to make sure spk_i always points at the 
              beginning of the appropriate memory segment. */
            sph_i[0] = q[0]*prefactors[0];  //l=0
            k = 1; ++sph_i;
            for (int l=1; l<l_max+1; l++) {            
                sph_i[l] = q[k]*prefactors[k];
                for (int m=1; m<l+1; m++) {
                    pq = q[k+m]*prefactors[k+m];
                    sph_i[l-m] = pq*s[m];
                    sph_i[l+m] = pq*c[m];
                }             
                k += l+1;
                sph_i += 2*l+1;
            }

            if (dsph != NULL) {
                // if the pointer is set, we compute the derivatives
                // we use the chain rule and some nice recursions

                // updates the pointer to the derivative storage
                double *dsph_i = dsph+i_sample*3*size_y; 

                
                k=0;
                // special case: l=0
                dsph_i[0] = dsph_i[size_y] = dsph_i[size_y*2] = 0;
                ++k; ++dsph_i;

                // special case: l=1
                dsph_i[1] = dsph_i[size_y+1] = 0;
                dsph_i[size_y*2+1] = prefactors[k]*1*q[k-1];
                ++k; 
                pq=prefactors[k]*q[k];
                dsph_i[0] = pq*s[0];
                dsph_i[2] = pq*c[0];
                dsph_i[size_y+0] = pq*c[0];
                dsph_i[size_y+2] = -pq*s[0];
                dsph_i[size_y*2+0] = 0;
                dsph_i[size_y*2+2] = 0;                
                ++k;                        
                dsph_i+=3;

                // general case - iteration
                for (int l=2; l<l_max+1; l++) {
                    dsph_i[l] = prefactors[k]*x*q[k-l+1];
                    dsph_i[size_y+l] = prefactors[k]*y*q[k-l+1];
                    dsph_i[size_y*2+l] = prefactors[k]*l*q[k-l];
                    
                    ++k;
                    for (int m=1; m<l-1; m++) {
                        // also includes a factor of m so we get the phi-dependent derivatives
                        pq=prefactors[k]*q[k]*m;  
                        pdq=prefactors[k]*q[k-l+1];
                        pdqx = pdq*x;
                        dsph_i[l-m] = (pdqx*s[m]+pq*s[m-1]);
                        dsph_i[l+m] = (pdqx*c[m]+pq*c[m-1]);
                        pdqy = pdq*y;
                        dsph_i[size_y+l-m] = (pdqy*s[m]+pq*c[m-1]);
                        dsph_i[size_y+l+m] = (pdqy*c[m]-pq*s[m-1]);
                        pdq=prefactors[k]*(l+m)*q[k-l];
                        dsph_i[size_y*2+l-m] = pdq*s[m];
                        dsph_i[size_y*2+l+m] = pdq*c[m];
                        ++k;
                    }

                    // do separately special cases that have lots of zeros
                    // m = l-1
                    pq=prefactors[k]*q[k]*(l-1); 
                    dsph_i[l-l+1] = pq*s[l-2];
                    dsph_i[l+l-1] = pq*c[l-2];
                    dsph_i[size_y+l-l+1] = pq*c[l-2];
                    dsph_i[size_y+l+l-1] = -pq*s[l-2];
                    pdq=prefactors[k]*(l+l-1)*q[k-l]; 
                    dsph_i[size_y*2+l-l+1] = pdq*s[l-1];
                    dsph_i[size_y*2+l+l-1] = pdq*c[l-1];
                    ++k;
                    //m=l
                    pq=prefactors[k]*q[k]*l; 
                    dsph_i[l-l] = pq*s[l-1];
                    dsph_i[l+l] = pq*c[l-1];
                    dsph_i[size_y+l-l] = pq*c[l-1];
                    dsph_i[size_y+l+l] = -pq*s[l-1];
                    dsph_i[size_y*2+l-l] = 0;
                    dsph_i[size_y*2+l+l] = 0;
                    ++k;
                    //advances the pointer for the sph derivatives
                    dsph_i += 2*l+1;  
                }
            }     
        }
        free(q);
        free(c);
        free(s);
    }
}

#define _HC_LMAX 5
void cartesian_spherical_harmonics(unsigned int n_samples, unsigned int l_max, 
            const double* prefactors, double *xyz, double *sph, double *dsph) {
    /*
        Computes "Cartesian" real spherical harmonics r^l*Y_lm(x,y,z) and 
        (optionally) their derivatives. This is an opinionated implementation:
        x,y,z are not scaled, and the resulting harmonic is scaled by r^l.
        This scaling allows for a stable, and fast implementation and the 
        r^l term can be easily incorporated into any radial function or 
        added a posteriori (with the corresponding derivative).
    */

    // call directly the fast ones
    if (l_max <= _HC_LMAX) {
        if (l_max==5) {
            cartesian_spherical_harmonics_l5(n_samples, xyz, sph, dsph);
        } else if (l_max == 4) {
            cartesian_spherical_harmonics_l4(n_samples, xyz, sph, dsph);
        } else if (l_max == 3) {
            cartesian_spherical_harmonics_l3(n_samples, xyz, sph, dsph);
        } else if (l_max == 2) {
            cartesian_spherical_harmonics_l2(n_samples, xyz, sph, dsph);
        } else if (l_max == 1) {
            cartesian_spherical_harmonics_l1(n_samples, xyz, sph, dsph);
        } else {
            cartesian_spherical_harmonics_l0(n_samples, xyz, sph, dsph);
        }
    } else {
    // general case, but start at 5
    #pragma omp parallel
    {
        // storage arrays for Qlm (modified associated Legendre polynomials)
        // and terms corresponding to (scaled) cosine and sine of the azimuth
        double* q = (double*) malloc(sizeof(double)*(l_max+1)*(l_max+2)/2);
        double* c = (double*) malloc(sizeof(double)*(l_max+1));
        double* s = (double*) malloc(sizeof(double)*(l_max+1));

        // temporaries to store prefactor*q and dq
        double pq, pdq, pdqx, pdqy; 
        int k, size_y = (l_max+1)*(l_max+1);

        // precompute the Qll's (that are constant)
        q[0+0] = 1.0;
        k=1;
        for (int l = 1; l < l_max+1; l++) {
            q[k+l] = -(2*l-1)*q[k-1];
            k += l+1; 
        }

        // also initialize the sine and cosine, these never change
        c[0] = 1.0;
        s[0] = 0.0;

        /* k is a utility index to traverse lm arrays. we store sph in 
           a contiguous dimension, with (lm)=[(00)(1-1)(10)(11)(2-2)(2-1)...]
           so we often write a nested loop on l and m and track where we
           got by incrementing a separate index k. */        
        #pragma omp for
        for (int i_sample=0; i_sample<n_samples; i_sample++) {

            double x = xyz[i_sample*3+0];
            double y = xyz[i_sample*3+1];
            double z = xyz[i_sample*3+2];
            double twoz = 2*z;
            double rxy = x*x+y*y;

            // pointer to the segment that should store the i_sample sph
            double *sph_i = sph+i_sample*size_y;
            
            /* These are scaled version of cos(m phi) and sin(m phi).
               Basically, these are cos and sin multiplied by r_xy^m,
               so that they are just plain polynomials of x,y,z.
            */
            for (int m = 1; m < l_max+1; m++) {
                c[m] = c[m-1]*x-s[m-1]*y;
                s[m] = c[m-1]*y+s[m-1]*x;
            }

            /* compute recursively the "Cartesian" associated Legendre polynomials Qlm.
               Qlm is defined as r^l/r_xy^m P_lm, and is a polynomial of x,y,z.
               These are computed with a recursive expression.
              */
            
            // Fills the Qlm starting from Qll (so we don't need to compute the low ls)
            // We need also Qlm for l=_HC_LMAX because of the derivatives
            // Initialize the recursion (Qll's are already stored and constant)
            k = (_HC_LMAX)*(_HC_LMAX+1)/2;
            for (int l = _HC_LMAX; l < l_max+1; l++) {
                q[k+(l-1)] = -z*q[k+l]; // (2*m-1)*z*q[k-1];
                k += l+1;                 
            }

            k = (_HC_LMAX)*(_HC_LMAX+1)/2;
            for (int l=_HC_LMAX; l < l_max+1; ++l) {
                double twomz = l*twoz;
                for (int m=l; m>=2; --m) {
                    twomz -= twoz;
                    q[k+m-2] = -(twomz*q[k+m-1]+rxy*q[k+m])/((l+(m-2)+1)*(l-(m-2)));                    
                }
                k += l+1;
            }
            
            /* fill the (Cartesian) sph by combining Qlm and 
              sine/cosine phi-dependent factors. we use pointer 
              arithmetics to make sure spk_i always points at the 
              beginning of the appropriate memory segment. */

            // these are the hard-coded, low-lmax
            _compute_sph_l0(sph_i);
            _compute_sph_l1(x,y,z,sph_i);
            double x2=x*x, y2=y*y, z2=z*z;
            _compute_sph_l2(x,y,z,x2,y2,z2,sph_i);
            _compute_sph_l3(x,y,z,x2,y2,z2,sph_i);
            _compute_sph_l4(x,y,z,x2,y2,z2,sph_i);
            _compute_sph_l5(x,y,z,x2,y2,z2,sph_i);

            k = (_HC_LMAX+1)*(_HC_LMAX+2)/2; sph_i += (_HC_LMAX+1)*(_HC_LMAX+1);
            for (int l=_HC_LMAX+1; l<l_max+1; l++) {            
                sph_i[l] = q[k]*prefactors[k];
                for (int m=1; m<l+1; m++) {
                    pq = q[k+m]*prefactors[k+m];
                    sph_i[l-m] = pq*s[m];
                    sph_i[l+m] = pq*c[m];
                }             
                k += l+1;
                sph_i += 2*l+1;
            }

            if (dsph != NULL) {
                // if the pointer is set, we compute the derivatives
                // we use the chain rule and some nice recursions

                // updates the pointer to the derivative storage
                double *dsph_i = dsph+i_sample*3*size_y; 
                sph_i = sph+i_sample*size_y; // must also re-set the sph pointer                

                _compute_dsph_l0(sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);
                _compute_dsph_l1(x,y,z,sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);                
                _compute_dsph_l2(x,y,z,x2,y2,z2,sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);
                _compute_dsph_l3(x,y,z,x2,y2,z2,sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);
                _compute_dsph_l4(x,y,z,x2,y2,z2,sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);
                _compute_dsph_l5(x,y,z,x2,y2,z2,sph_i,dsph_i,dsph_i+size_y,dsph_i+size_y*2);
                
                k = (_HC_LMAX+1)*(_HC_LMAX+2)/2; dsph_i += (_HC_LMAX+1)*(_HC_LMAX+1);
                // general case - iteration
                for (int l=_HC_LMAX+1; l<l_max+1; l++) {
                    dsph_i[l] = prefactors[k]*x*q[k-l+1];
                    dsph_i[size_y+l] = prefactors[k]*y*q[k-l+1];
                    dsph_i[size_y*2+l] = prefactors[k]*l*q[k-l];
                    
                    ++k;
                    for (int m=1; m<l-1; m++) {
                        // also includes a factor of m so we get the phi-dependent derivatives
                        pq=prefactors[k]*q[k]*m;  
                        pdq=prefactors[k]*q[k-l+1];
                        pdqx = pdq*x;
                        dsph_i[l-m] = (pdqx*s[m]+pq*s[m-1]);
                        dsph_i[l+m] = (pdqx*c[m]+pq*c[m-1]);
                        pdqy = pdq*y;
                        dsph_i[size_y+l-m] = (pdqy*s[m]+pq*c[m-1]);
                        dsph_i[size_y+l+m] = (pdqy*c[m]-pq*s[m-1]);
                        pdq=prefactors[k]*(l+m)*q[k-l];
                        dsph_i[size_y*2+l-m] = pdq*s[m];
                        dsph_i[size_y*2+l+m] = pdq*c[m];
                        ++k;
                    }

                    // do separately special cases that have lots of zeros
                    // m = l-1
                    pq=prefactors[k]*q[k]*(l-1); 
                    dsph_i[l-l+1] = pq*s[l-2];
                    dsph_i[l+l-1] = pq*c[l-2];
                    dsph_i[size_y+l-l+1] = pq*c[l-2];
                    dsph_i[size_y+l+l-1] = -pq*s[l-2];
                    pdq=prefactors[k]*(l+l-1)*q[k-l]; 
                    dsph_i[size_y*2+l-l+1] = pdq*s[l-1];
                    dsph_i[size_y*2+l+l-1] = pdq*c[l-1];
                    ++k;
                    //m=l
                    pq=prefactors[k]*q[k]*l; 
                    dsph_i[l-l] = pq*s[l-1];
                    dsph_i[l+l] = pq*c[l-1];
                    dsph_i[size_y+l-l] = pq*c[l-1];
                    dsph_i[size_y+l+l] = -pq*s[l-1];
                    dsph_i[size_y*2+l-l] = 0;
                    dsph_i[size_y*2+l+l] = 0;
                    ++k;
                    //advances the pointer for the sph derivatives
                    dsph_i += 2*l+1;  
                }
            }     
        }
        free(q);
        free(c);
        free(s);
    }
    }
}
