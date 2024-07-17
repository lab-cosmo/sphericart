#include <cmath>
#include <stdexcept>

#define _SPHERICART_INTERNAL_IMPLEMENTATION
#include "sphericart.hpp"

using namespace sphericart;

// This macro defines the different possible hardcoded function calls. It is
// used to initialize the function pointers that are used by the `compute_`
// calls in the SphericalHarmonics class
#define _HARCODED_SWITCH_CASE(L_MAX)                                                               \
    if (this->normalized) {                                                                        \
        this->_array_no_derivatives = &hardcoded_sph<T, false, false, true, L_MAX>;                \
        this->_array_with_derivatives = &hardcoded_sph<T, true, false, true, L_MAX>;               \
        this->_sample_no_derivatives = &hardcoded_sph_sample<T, false, false, true, L_MAX>;        \
        this->_sample_with_derivatives = &hardcoded_sph_sample<T, true, false, true, L_MAX>;       \
    } else {                                                                                       \
        this->_array_no_derivatives = &hardcoded_sph<T, false, false, false, L_MAX>;               \
        this->_array_with_derivatives = &hardcoded_sph<T, true, false, false, L_MAX>;              \
        this->_sample_no_derivatives = &hardcoded_sph_sample<T, false, false, false, L_MAX>;       \
        this->_sample_with_derivatives = &hardcoded_sph_sample<T, true, false, false, L_MAX>;      \
    }

template <typename T> SphericalHarmonics<T>::SphericalHarmonics(size_t l_max, bool normalized) {
    /*
        This is the constructor of the SphericalHarmonics class. It initizlizes
       buffer space, compute prefactors, and sets the function pointers that are
       used for the actual calls
    */

    this->l_max = (int)l_max;
    this->size_y = (int)(l_max + 1) * (l_max + 1);
    this->size_q = (int)(l_max + 1) * (l_max + 2) / 2;
    this->normalized = normalized;
    this->prefactors = new T[this->size_q * 2];
    this->omp_num_threads = omp_get_max_threads();

    // buffers for cos, sin, 2mz arrays
    // allocates buffers that are large enough to store thread-local data
    this->buffers = new T[this->size_q * 3 * this->omp_num_threads];

    compute_sph_prefactors<T>((int)l_max, this->prefactors);

    // sets the correct function pointers for the compute functions
    if (this->l_max <= SPHERICART_LMAX_HARDCODED) {
        // If we only need hard-coded calls, we set them up at this point
        // using a macro to avoid even more code duplication.
        switch (this->l_max) {
        case 0:
            _HARCODED_SWITCH_CASE(0);
            break;
        case 1:
            _HARCODED_SWITCH_CASE(1);
            break;
        case 2:
            _HARCODED_SWITCH_CASE(2);
            break;
        case 3:
            _HARCODED_SWITCH_CASE(3);
            break;
        case 4:
            _HARCODED_SWITCH_CASE(4);
            break;
        case 5:
            _HARCODED_SWITCH_CASE(5);
            break;
        case 6:
            _HARCODED_SWITCH_CASE(6);
            break;
        }
    } else {
        if (this->normalized) {
            this->_array_no_derivatives =
                &generic_sph<T, false, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives =
                &generic_sph<T, true, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives =
                &generic_sph_sample<T, false, false, true, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives =
                &generic_sph_sample<T, true, false, true, SPHERICART_LMAX_HARDCODED>;
        } else {
            this->_array_no_derivatives =
                &generic_sph<T, false, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_array_with_derivatives =
                &generic_sph<T, true, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_no_derivatives =
                &generic_sph_sample<T, false, false, false, SPHERICART_LMAX_HARDCODED>;
            this->_sample_with_derivatives =
                &generic_sph_sample<T, true, false, false, SPHERICART_LMAX_HARDCODED>;
        }
    }

    // set up the second derivative functions
    if (this->normalized) {
        if (this->l_max == 0) {
            this->_array_with_hessians = &hardcoded_sph<T, true, true, true, 0>;
            this->_sample_with_hessians = &hardcoded_sph_sample<T, true, true, true, 0>;
        } else if (this->l_max == 1) {
            this->_array_with_hessians = &hardcoded_sph<T, true, true, true, 1>;
            this->_sample_with_hessians = &hardcoded_sph_sample<T, true, true, true, 1>;
        } else { // second derivatives are not hardcoded past l = 1. Call
                 // generic
            // implementations
            this->_array_with_hessians = &generic_sph<T, true, true, true, 1>;
            this->_sample_with_hessians = &generic_sph_sample<T, true, true, true, 1>;
        }
    } else {
        if (this->l_max == 0) {
            this->_array_with_hessians = &hardcoded_sph<T, true, true, false, 0>;
            this->_sample_with_hessians = &hardcoded_sph_sample<T, true, true, false, 0>;
        } else if (this->l_max == 1) {
            this->_array_with_hessians = &hardcoded_sph<T, true, true, false, 1>;
            this->_sample_with_hessians = &hardcoded_sph_sample<T, true, true, false, 1>;
        } else { // second derivatives are not hardcoded past l = 1. Call
                 // generic
            // implementations
            this->_array_with_hessians = &generic_sph<T, true, true, false, 1>;
            this->_sample_with_hessians = &generic_sph_sample<T, true, true, false, 1>;
        }
    }
}

template <typename T> SphericalHarmonics<T>::~SphericalHarmonics() {
    // Destructor, simply frees buffers
    delete[] this->prefactors;
    delete[] this->buffers;
}

// The compute/compute_with_gradient functions decide which function to call
// based on the size of the input vectors

template <typename T>
void SphericalHarmonics<T>::compute(const std::vector<T>& xyz, std::vector<T>& sph) {
    auto n_samples = xyz.size() / 3;
    sph.resize(n_samples * (l_max + 1) * (l_max + 1));

    if (xyz.size() == 3) {
        this->compute_sample(xyz.data(), xyz.size(), sph.data(), sph.size());
    } else {
        this->compute_array(xyz.data(), xyz.size(), sph.data(), sph.size());
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_with_gradients(
    const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph
) {
    auto n_samples = xyz.size() / 3;
    sph.resize(n_samples * (l_max + 1) * (l_max + 1));
    dsph.resize(n_samples * 3 * (l_max + 1) * (l_max + 1));

    if (xyz.size() == 3) {
        this->compute_sample_with_gradients(
            xyz.data(), xyz.size(), sph.data(), sph.size(), dsph.data(), dsph.size()
        );
    } else {
        this->compute_array_with_gradients(
            xyz.data(), xyz.size(), sph.data(), sph.size(), dsph.data(), dsph.size()
        );
    }
}

// Lower-level functions are simply wrappers over function-pointer-style calls
// that invoke the right (hardcoded or generic) C++-library call
template <typename T>
void SphericalHarmonics<T>::compute_with_hessians(
    const std::vector<T>& xyz, std::vector<T>& sph, std::vector<T>& dsph, std::vector<T>& ddsph
) {
    auto n_samples = xyz.size() / 3;
    sph.resize(n_samples * (l_max + 1) * (l_max + 1));
    dsph.resize(n_samples * 3 * (l_max + 1) * (l_max + 1));
    ddsph.resize(n_samples * 9 * (l_max + 1) * (l_max + 1));

    if (xyz.size() == 3) {
        this->compute_sample_with_hessians(
            xyz.data(),
            xyz.size(),
            sph.data(),
            sph.size(),
            dsph.data(),
            dsph.size(),
            ddsph.data(),
            ddsph.size()
        );
    } else {
        this->compute_array_with_hessians(
            xyz.data(),
            xyz.size(),
            sph.data(),
            sph.size(),
            dsph.data(),
            dsph.size(),
            ddsph.data(),
            ddsph.size()
        );
    }
}

template <typename T>
void SphericalHarmonics<T>::compute_array(const T* xyz, size_t xyz_length, T* sph, size_t sph_length) {
    if (xyz_length % 3 != 0) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "xyz array with `n_samples "
                                 "x 3` elements");
    }

    auto n_samples = xyz_length / 3;
    if (n_samples == 0) {
        // nothing to compute; we return here because some libraries (e.g. torch)
        // seem to use nullptrs for tensors with 0 elements
        return;
    }
    if (sph == nullptr || sph_length < (n_samples * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "sph array with `n_samples "
                                 "x (l_max + 1)^2` elements");
    }

    this->_array_no_derivatives(
        xyz, sph, nullptr, nullptr, n_samples, this->l_max, this->prefactors, this->buffers
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_array_with_gradients(
    const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length
) {
    if (xyz_length % 3 != 0) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "xyz array with `n_samples "
                                 "x 3` elements");
    }

    auto n_samples = xyz_length / 3;
    if (n_samples == 0) {
        // nothing to compute; we return here because some libraries (e.g. torch)
        // seem to use nullptrs for tensors with 0 elements
        return;
    }
    if (sph == nullptr || sph_length < (n_samples * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "sph array with `n_samples "
                                 "x (l_max + 1)^2` elements");
    }

    if (dsph == nullptr || dsph_length < (n_samples * 3 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected dsph array with "
                                 "`n_samples x 3 x (l_max + 1)^2` elements");
    }

    this->_array_with_derivatives(
        xyz, sph, dsph, nullptr, n_samples, this->l_max, this->prefactors, this->buffers
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_array_with_hessians(
    const T* xyz,
    size_t xyz_length,
    T* sph,
    size_t sph_length,
    T* dsph,
    size_t dsph_length,
    T* ddsph,
    size_t ddsph_length
) {
    if (xyz_length % 3 != 0) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "xyz array with `n_samples "
                                 "x 3` elements");
    }

    auto n_samples = xyz_length / 3;
    if (n_samples == 0) {
        // nothing to compute; we return here because some libraries (e.g. torch)
        // seem to use nullptrs for tensors with 0 elements
        return;
    }
    if (sph == nullptr || sph_length < (n_samples * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected "
                                 "sph array with `n_samples "
                                 "x (l_max + 1)^2` elements");
    }

    if (dsph == nullptr || dsph_length < (n_samples * 3 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected dsph array with "
                                 "`n_samples x 3 x (l_max + 1)^2` elements");
    }

    if (ddsph == nullptr || ddsph_length < (n_samples * 9 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_array: expected ddsph array with "
                                 "`n_samples x 9 x (l_max + 1)^2` elements");
    }

    this->_array_with_hessians(
        xyz, sph, dsph, ddsph, n_samples, this->l_max, this->prefactors, this->buffers
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_sample(const T* xyz, size_t xyz_length, T* sph, size_t sph_length) {
    if (xyz_length != 3) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected xyz array with 3 "
                                 "elements");
    }

    if (sph == nullptr || sph_length < ((l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected "
                                 "sph array with `(l_max + "
                                 "1)^2` elements");
    }

    this->_sample_no_derivatives(
        xyz,
        sph,
        nullptr,
        nullptr,
        this->l_max,
        this->size_y,
        this->prefactors,
        this->prefactors + this->size_q,
        this->buffers,
        this->buffers + this->size_q,
        this->buffers + 2 * this->size_q
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_sample_with_gradients(
    const T* xyz, size_t xyz_length, T* sph, size_t sph_length, T* dsph, size_t dsph_length
) {
    if (xyz_length != 3) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected xyz array with 3 "
                                 "elements");
    }

    if (sph == nullptr || sph_length < ((l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected "
                                 "sph array with `(l_max + "
                                 "1)^2` elements");
    }

    if (dsph == nullptr || dsph_length < (3 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected sph array with `3 x "
                                 "(l_max + 1)^2` elements");
    }

    this->_sample_with_derivatives(
        xyz,
        sph,
        dsph,
        nullptr,
        this->l_max,
        this->size_y,
        this->prefactors,
        this->prefactors + this->size_q,
        this->buffers,
        this->buffers + this->size_q,
        this->buffers + 2 * this->size_q
    );
}

template <typename T>
void SphericalHarmonics<T>::compute_sample_with_hessians(
    const T* xyz,
    size_t xyz_length,
    T* sph,
    size_t sph_length,
    T* dsph,
    size_t dsph_length,
    T* ddsph,
    size_t ddsph_length
) {
    if (xyz_length != 3) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected xyz array with 3 "
                                 "elements");
    }

    if (sph == nullptr || sph_length < ((l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected "
                                 "sph array with `(l_max + "
                                 "1)^2` elements");
    }

    if (dsph == nullptr || dsph_length < (3 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error(
            "SphericalHarmonics::compute_sample: expected dsph array with `3 x "
            "(l_max + 1)^2` elements"
        );
    }

    if (ddsph == nullptr || ddsph_length < (9 * (l_max + 1) * (l_max + 1))) {
        throw std::runtime_error("SphericalHarmonics::compute_sample: expected "
                                 "ddsph array with `9 x "
                                 "(l_max + 1)^2` elements");
    }

    this->_sample_with_hessians(
        xyz,
        sph,
        dsph,
        ddsph,
        this->l_max,
        this->size_y,
        this->prefactors,
        this->prefactors + this->size_q,
        this->buffers,
        this->buffers + this->size_q,
        this->buffers + 2 * this->size_q
    );
}

// instantiates the SphericalHarmonics class for basic floating point types
template class sphericart::SphericalHarmonics<float>;
template class sphericart::SphericalHarmonics<double>;
