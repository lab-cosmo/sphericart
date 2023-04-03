#ifndef SPHERICART_OPENMP_SUPPORT_HPP
#define SPHERICART_OPENMP_SUPPORT_HPP

#ifdef _OPENMP

    #include <omp.h>

#else
    // define dummy versions of the functions we need

    static inline int omp_get_max_threads() {
        return 1;
    }

    static inline int omp_get_thread_num() {
        return 0;
    }

#endif


#endif
