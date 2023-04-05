#include "sphericart.hpp"

#include "sphericart/autograd.hpp"
#include "sphericart/torch.hpp"
#include "sphericart/cuda.hpp"

using namespace sphericart_torch;

std::vector<torch::Tensor> SphericalHarmonics::compute_raw_cpu(torch::Tensor xyz, bool do_gradients) {
    if (!xyz.is_contiguous()) {
        throw std::runtime_error("this code only runs with contiguous tensors");
    }

    if (!xyz.device().is_cpu()) {
        throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
    }

    auto n_samples = xyz.sizes()[0];
    auto options = torch::TensorOptions().device(xyz.device()).dtype(xyz.dtype());

    auto sph_length = n_samples * (l_max_ + 1) * (l_max_ + 1);
    auto dsph_length = n_samples * 3 * (l_max_ + 1) * (l_max_ + 1);
    auto sph = torch::zeros({n_samples, (l_max_ + 1) * (l_max_ + 1)}, options);

    if (xyz.dtype() == c10::kDouble) {
        if (do_gradients) {
            auto dsph = torch::zeros({n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);

            calculator_double_.compute_array(
                xyz.data_ptr<double>(),
                n_samples * 3,
                sph.data_ptr<double>(),
                sph_length,
                dsph.data_ptr<double>(),
                dsph_length
            );

            return {sph, dsph};
        } else {
            calculator_double_.compute_array(
                xyz.data_ptr<double>(),
                n_samples * 3,
                sph.data_ptr<double>(),
                sph_length
            );
            return {sph, torch::Tensor()};
        }
    } else if (xyz.dtype() == c10::kFloat) {
        if (do_gradients) {
            auto dsph = torch::zeros({n_samples, 3, (l_max_ + 1) * (l_max_ + 1)}, options);
            calculator_float_.compute_array(
                xyz.data_ptr<float>(),
                n_samples * 3,
                sph.data_ptr<float>(),
                sph_length,
                dsph.data_ptr<float>(),
                dsph_length
            );
            return {sph, dsph};
        } else {
            calculator_float_.compute_array(
                xyz.data_ptr<float>(),
                n_samples * 3,
                sph.data_ptr<float>(),
                sph_length
            );
            return {sph, torch::Tensor()};
        }
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }
}

static torch::Tensor backward_cpu(torch::Tensor xyz, torch::Tensor dsph, torch::Tensor sph_grad) {
    auto xyz_grad = torch::zeros_like(xyz);

    if (!sph_grad.device().is_cpu() || !xyz.device().is_cpu() || !dsph.device().is_cpu()) {
        throw std::runtime_error("internal error: called CPU version on non-CPU tensor");
    }

    // we need contiguous data to take pointers below
    sph_grad = sph_grad.contiguous();

    if (!xyz_grad.is_contiguous() || !dsph.is_contiguous()) {
        // we created these, they should always be contiguous
        throw std::runtime_error("internal error: xyz_grad or dsph are not contiguous");
    }

    auto n_samples = xyz.sizes()[0];
    auto n_sph = sph_grad.sizes()[1];
    if (xyz.dtype() == c10::kDouble) {
        auto xyz_grad_p = xyz_grad.data_ptr<double>();
        auto sph_grad_p = sph_grad.data_ptr<double>();
        auto dsph_p = dsph.data_ptr<double>();

        #pragma omp parallel for
        for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
            for (size_t spatial=0; spatial<3; spatial++) {
                for (int i_sph=0; i_sph<n_sph; i_sph++) {
                    xyz_grad_p[3*i_sample+spatial] += sph_grad_p[n_sph*i_sample+i_sph] * dsph_p[n_sph*3*i_sample+n_sph*spatial+i_sph];
                }
            }
        }
    } else if (xyz.dtype() == c10::kFloat) {
        auto xyz_grad_p = xyz_grad.data_ptr<float>();
        auto sph_grad_p = sph_grad.data_ptr<float>();
        auto dsph_p = dsph.data_ptr<float>();

        #pragma omp parallel for
        for (size_t i_sample=0; i_sample<n_samples; i_sample++) {
            for (size_t spatial=0; spatial<3; spatial++) {
                for (int i_sph=0; i_sph<n_sph; i_sph++) {
                    xyz_grad_p[3*i_sample+spatial] += sph_grad_p[n_sph*i_sample+i_sph] * dsph_p[n_sph*3*i_sample+n_sph*spatial+i_sph];
                }
            }
        }
    } else {
        throw std::runtime_error("this code only runs on float64 and float32 arrays");
    }

    return xyz_grad;
}

/* ===========================================================================*/

bool CudaSharedMemorySettings::update_if_required(
    torch::ScalarType scalar_type,
    int64_t l_max,
    int64_t GRID_DIM_X,
    int64_t GRID_DIM_Y,
    bool gradients
) {
    auto scalar_size = torch::elementSize(scalar_type);
    if (this->l_max_ >= l_max &&
        this->grid_dim_x_ >= GRID_DIM_X &&
        this->grid_dim_y_ >= GRID_DIM_Y &&
        this->scalar_size_ >= scalar_size &&
        (this->requires_grad_ || !gradients)
    ) {
        // no need to adjust shared memory
        return true;
    }

    bool result = adjust_cuda_shared_memory(scalar_type, l_max, GRID_DIM_X, GRID_DIM_Y, gradients);

    if (result){
        this->l_max_ = l_max;
        this->grid_dim_x_ = GRID_DIM_X;
        this->grid_dim_y_ = GRID_DIM_Y;
        this->requires_grad_ = gradients;
        this->scalar_size_ = scalar_size;
    }
    return result;
}

/* ===========================================================================*/

torch::autograd::variable_list SphericalHarmonicsAutograd::forward(
    torch::autograd::AutogradContext *ctx,
    SphericalHarmonics& calculator,
    torch::Tensor xyz,
    bool gradients
) {
    if (xyz.sizes().size() != 2) {
        throw std::runtime_error("xyz tensor must be a 2D array");
    }

    if (xyz.sizes()[1] != 3) {
        throw std::runtime_error("xyz tensor must be an `n_samples x 3` array");
    }

    auto sph = torch::Tensor();
    auto dsph = torch::Tensor();

    if (xyz.device().is_cpu()) {
        auto results = calculator.compute_raw_cpu(xyz, gradients || xyz.requires_grad());
        sph = results[0];
        dsph = results[1];
    } else if (xyz.device().is_cuda()) {
        // re-do the shared memory update in case `requires_grad` changed
        const int GRID_DIM_Y = 1;
        int GRID_DIM_X = 32;

        int dtype = torch::elementSize(xyz.scalar_type());

        const std::lock_guard<std::mutex> guard(calculator.cuda_shmem_mutex_);
        
        bool shm_result = calculator.cuda_shmem_.update_if_required(
            xyz.scalar_type(),
            calculator.l_max_,
            GRID_DIM_X,
            GRID_DIM_Y,
            xyz.requires_grad()||gradients
        );
        
        if (!shm_result){
            printf("Warning: Failed to update shared memory specification with element_size = %d, GRID_DIM_X = %d, GRID_DIM_Y = %d, xyz.requires_grad() || gradients = %s\n",
                    dtype, GRID_DIM_X, GRID_DIM_Y, xyz.requires_grad()||gradients ? "true" : "false");
            printf ("Re-attempting with GRID_DIM_X= 16\n");

            GRID_DIM_X = 16;
            shm_result = calculator.cuda_shmem_.update_if_required(
                xyz.scalar_type(),
                calculator.l_max_,
                GRID_DIM_X,
                GRID_DIM_Y,
                xyz.requires_grad()||gradients
            );

            if (!shm_result) {
                throw std::runtime_error("Insufficient shared memory available to compute spherical_harmonics with requested parameters.");
            } else {
                printf("shared memory update OK.\n");
            }
        }

        auto prefactors = torch::Tensor();
        if (xyz.dtype() == c10::kDouble) {
            prefactors = calculator.prefactors_cuda_double_;
        } else if (xyz.dtype() == c10::kFloat) {
            prefactors = calculator.prefactors_cuda_float_;
        } else {
            throw std::runtime_error("this code only runs on float64 and float32 arrays");
        }

        auto results = spherical_harmonics_cuda(
            xyz,
            prefactors,
            calculator.l_max_,
            calculator.normalize_,
            GRID_DIM_X,
            GRID_DIM_Y,
	    gradients
        );
        sph = results[0];
        dsph = results[1];
	//printf("Computed CUDA with derivatives %f\n", dsph[5]);
    } else {
        throw std::runtime_error("Spherical harmonics are only implemented for CPU and CUDA");
    }

    if (xyz.requires_grad()) {
        ctx->save_for_backward({xyz, dsph});
    }

    if (gradients) {
        return {sph, dsph};
    } else {
        return {sph};
    }

}

torch::autograd::variable_list SphericalHarmonicsAutograd::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs
) {
    /* get the saved data from the forward pass */
    auto saved_variables = ctx->get_saved_variables();
    auto xyz = saved_variables[0];
    auto dsph = saved_variables[1];

    auto sph_grad = grad_outputs[0];

    auto xyz_grad = torch::Tensor();
    if (xyz.requires_grad()) {
        if (xyz.device().is_cpu()) {
            xyz_grad = backward_cpu(xyz, dsph, sph_grad);
        } else if (xyz.device().is_cuda()) {
            xyz_grad = spherical_harmonics_backward_cuda(xyz, dsph, sph_grad);
        } else {
            throw std::runtime_error("Spherical harmonics are only implemented for CPU and CUDA");
        }
    }

    if (grad_outputs.size() > 1) {
        throw std::runtime_error("We can not run a backward pass through the gradients of spherical harmonics");
    }

    return {torch::Tensor(), xyz_grad, torch::Tensor()};
}
