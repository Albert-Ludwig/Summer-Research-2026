/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

#ifndef ACADOS_SOLVER_diffusion_projection_unicycle_H_
#define ACADOS_SOLVER_diffusion_projection_unicycle_H_

#include "acados/utils/types.h"

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#define DIFFUSION_PROJECTION_UNICYCLE_NX     5
#define DIFFUSION_PROJECTION_UNICYCLE_NZ     0
#define DIFFUSION_PROJECTION_UNICYCLE_NU     2
#define DIFFUSION_PROJECTION_UNICYCLE_NP     90
#define DIFFUSION_PROJECTION_UNICYCLE_NP_GLOBAL     0
#define DIFFUSION_PROJECTION_UNICYCLE_NBX    2
#define DIFFUSION_PROJECTION_UNICYCLE_NBX0   5
#define DIFFUSION_PROJECTION_UNICYCLE_NBU    2
#define DIFFUSION_PROJECTION_UNICYCLE_NSBX   0
#define DIFFUSION_PROJECTION_UNICYCLE_NSBU   0
#define DIFFUSION_PROJECTION_UNICYCLE_NSH    30
#define DIFFUSION_PROJECTION_UNICYCLE_NSH0   0
#define DIFFUSION_PROJECTION_UNICYCLE_NSG    0
#define DIFFUSION_PROJECTION_UNICYCLE_NSPHI  0
#define DIFFUSION_PROJECTION_UNICYCLE_NSHN   30
#define DIFFUSION_PROJECTION_UNICYCLE_NSGN   0
#define DIFFUSION_PROJECTION_UNICYCLE_NSPHIN 0
#define DIFFUSION_PROJECTION_UNICYCLE_NSPHI0 0
#define DIFFUSION_PROJECTION_UNICYCLE_NSBXN  0
#define DIFFUSION_PROJECTION_UNICYCLE_NS     30
#define DIFFUSION_PROJECTION_UNICYCLE_NS0    0
#define DIFFUSION_PROJECTION_UNICYCLE_NSN    30
#define DIFFUSION_PROJECTION_UNICYCLE_NG     0
#define DIFFUSION_PROJECTION_UNICYCLE_NBXN   0
#define DIFFUSION_PROJECTION_UNICYCLE_NGN    0
#define DIFFUSION_PROJECTION_UNICYCLE_NY0    4
#define DIFFUSION_PROJECTION_UNICYCLE_NY     4
#define DIFFUSION_PROJECTION_UNICYCLE_NYN    2
#define DIFFUSION_PROJECTION_UNICYCLE_N      31
#define DIFFUSION_PROJECTION_UNICYCLE_NH     30
#define DIFFUSION_PROJECTION_UNICYCLE_NHN    30
#define DIFFUSION_PROJECTION_UNICYCLE_NH0    0
#define DIFFUSION_PROJECTION_UNICYCLE_NPHI0  0
#define DIFFUSION_PROJECTION_UNICYCLE_NPHI   0
#define DIFFUSION_PROJECTION_UNICYCLE_NPHIN  0
#define DIFFUSION_PROJECTION_UNICYCLE_NR     0

#ifdef __cplusplus
extern "C" {
#endif


// ** capsule for solver data **
typedef struct diffusion_projection_unicycle_solver_capsule
{
    // acados objects
    ocp_nlp_in *nlp_in;
    ocp_nlp_out *nlp_out;
    ocp_nlp_out *sens_out;
    ocp_nlp_solver *nlp_solver;
    void *nlp_opts;
    ocp_nlp_plan_t *nlp_solver_plan;
    ocp_nlp_config *nlp_config;
    ocp_nlp_dims *nlp_dims;

    // number of expected runtime parameters
    unsigned int nlp_np;

    /* external functions */

    // dynamics

    external_function_external_param_casadi *expl_vde_forw;
    external_function_external_param_casadi *expl_vde_forw_p;
    external_function_external_param_casadi *expl_ode_fun;
    external_function_external_param_casadi *expl_vde_adj;




    // cost






    // constraints
    external_function_external_param_casadi *nl_constr_h_fun_jac;
    external_function_external_param_casadi *nl_constr_h_fun;








    external_function_external_param_casadi nl_constr_h_e_fun_jac;
    external_function_external_param_casadi nl_constr_h_e_fun;



} diffusion_projection_unicycle_solver_capsule;

ACADOS_SYMBOL_EXPORT diffusion_projection_unicycle_solver_capsule * diffusion_projection_unicycle_acados_create_capsule(void);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_free_capsule(diffusion_projection_unicycle_solver_capsule *capsule);

ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_create(diffusion_projection_unicycle_solver_capsule * capsule);

ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_reset(diffusion_projection_unicycle_solver_capsule* capsule, int reset_qp_solver_mem);

/**
 * Generic version of diffusion_projection_unicycle_acados_create which allows to use a different number of shooting intervals than
 * the number used for code generation. If new_time_steps=NULL and n_time_steps matches the number used for code
 * generation, the time-steps from code generation is used.
 */
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_create_with_discretization(diffusion_projection_unicycle_solver_capsule * capsule, int n_time_steps, double* new_time_steps);
/**
 * Update the time step vector. Number N must be identical to the currently set number of shooting nodes in the
 * nlp_solver_plan. Returns 0 if no error occurred and a otherwise a value other than 0.
 */
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_update_time_steps(diffusion_projection_unicycle_solver_capsule * capsule, int N, double* new_time_steps);
/**
 * This function is used for updating an already initialized solver with a different number of qp_cond_N.
 */
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_update_qp_solver_cond_N(diffusion_projection_unicycle_solver_capsule * capsule, int qp_solver_cond_N);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_update_params(diffusion_projection_unicycle_solver_capsule * capsule, int stage, double *value, int np);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_update_params_sparse(diffusion_projection_unicycle_solver_capsule * capsule, int stage, int *idx, double *p, int n_update);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_set_p_global_and_precompute_dependencies(diffusion_projection_unicycle_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_solve(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_setup_qp_matrices_and_factorize(diffusion_projection_unicycle_solver_capsule* capsule);



ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_free(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void diffusion_projection_unicycle_acados_print_stats(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT int diffusion_projection_unicycle_acados_custom_update(diffusion_projection_unicycle_solver_capsule* capsule, double* data, int data_len);

ACADOS_SYMBOL_EXPORT ocp_nlp_in *diffusion_projection_unicycle_acados_get_nlp_in(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *diffusion_projection_unicycle_acados_get_nlp_out(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_out *diffusion_projection_unicycle_acados_get_sens_out(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_solver *diffusion_projection_unicycle_acados_get_nlp_solver(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_config *diffusion_projection_unicycle_acados_get_nlp_config(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT void *diffusion_projection_unicycle_acados_get_nlp_opts(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_dims *diffusion_projection_unicycle_acados_get_nlp_dims(diffusion_projection_unicycle_solver_capsule * capsule);
ACADOS_SYMBOL_EXPORT ocp_nlp_plan_t *diffusion_projection_unicycle_acados_get_nlp_plan(diffusion_projection_unicycle_solver_capsule * capsule);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // ACADOS_SOLVER_diffusion_projection_unicycle_H_
