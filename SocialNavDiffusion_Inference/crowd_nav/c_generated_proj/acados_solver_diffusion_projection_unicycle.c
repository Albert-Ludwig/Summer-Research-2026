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

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific

#include "diffusion_projection_unicycle_model/diffusion_projection_unicycle_model.h"


#include "diffusion_projection_unicycle_constraints/diffusion_projection_unicycle_constraints.h"



#include "acados_solver_diffusion_projection_unicycle.h"

#define NX     DIFFUSION_PROJECTION_UNICYCLE_NX
#define NZ     DIFFUSION_PROJECTION_UNICYCLE_NZ
#define NU     DIFFUSION_PROJECTION_UNICYCLE_NU
#define NP     DIFFUSION_PROJECTION_UNICYCLE_NP
#define NP_GLOBAL     DIFFUSION_PROJECTION_UNICYCLE_NP_GLOBAL
#define NY0    DIFFUSION_PROJECTION_UNICYCLE_NY0
#define NY     DIFFUSION_PROJECTION_UNICYCLE_NY
#define NYN    DIFFUSION_PROJECTION_UNICYCLE_NYN

#define NBX    DIFFUSION_PROJECTION_UNICYCLE_NBX
#define NBX0   DIFFUSION_PROJECTION_UNICYCLE_NBX0
#define NBU    DIFFUSION_PROJECTION_UNICYCLE_NBU
#define NG     DIFFUSION_PROJECTION_UNICYCLE_NG
#define NBXN   DIFFUSION_PROJECTION_UNICYCLE_NBXN
#define NGN    DIFFUSION_PROJECTION_UNICYCLE_NGN

#define NH     DIFFUSION_PROJECTION_UNICYCLE_NH
#define NHN    DIFFUSION_PROJECTION_UNICYCLE_NHN
#define NH0    DIFFUSION_PROJECTION_UNICYCLE_NH0
#define NPHI   DIFFUSION_PROJECTION_UNICYCLE_NPHI
#define NPHIN  DIFFUSION_PROJECTION_UNICYCLE_NPHIN
#define NPHI0  DIFFUSION_PROJECTION_UNICYCLE_NPHI0
#define NR     DIFFUSION_PROJECTION_UNICYCLE_NR

#define NS     DIFFUSION_PROJECTION_UNICYCLE_NS
#define NS0    DIFFUSION_PROJECTION_UNICYCLE_NS0
#define NSN    DIFFUSION_PROJECTION_UNICYCLE_NSN

#define NSBX   DIFFUSION_PROJECTION_UNICYCLE_NSBX
#define NSBU   DIFFUSION_PROJECTION_UNICYCLE_NSBU
#define NSH0   DIFFUSION_PROJECTION_UNICYCLE_NSH0
#define NSH    DIFFUSION_PROJECTION_UNICYCLE_NSH
#define NSHN   DIFFUSION_PROJECTION_UNICYCLE_NSHN
#define NSG    DIFFUSION_PROJECTION_UNICYCLE_NSG
#define NSPHI0 DIFFUSION_PROJECTION_UNICYCLE_NSPHI0
#define NSPHI  DIFFUSION_PROJECTION_UNICYCLE_NSPHI
#define NSPHIN DIFFUSION_PROJECTION_UNICYCLE_NSPHIN
#define NSGN   DIFFUSION_PROJECTION_UNICYCLE_NSGN
#define NSBXN  DIFFUSION_PROJECTION_UNICYCLE_NSBXN



// ** solver data **

diffusion_projection_unicycle_solver_capsule * diffusion_projection_unicycle_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(diffusion_projection_unicycle_solver_capsule));
    diffusion_projection_unicycle_solver_capsule *capsule = (diffusion_projection_unicycle_solver_capsule *) capsule_mem;

    return capsule;
}


int diffusion_projection_unicycle_acados_free_capsule(diffusion_projection_unicycle_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int diffusion_projection_unicycle_acados_create(diffusion_projection_unicycle_solver_capsule* capsule)
{
    int N_shooting_intervals = DIFFUSION_PROJECTION_UNICYCLE_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return diffusion_projection_unicycle_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int diffusion_projection_unicycle_acados_update_time_steps(diffusion_projection_unicycle_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "diffusion_projection_unicycle_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 1
 */
void diffusion_projection_unicycle_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = LINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = LINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* diffusion_projection_unicycle_acados_create_setup_dimensions(diffusion_projection_unicycle_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 5;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 3
 */
void diffusion_projection_unicycle_acados_create_setup_functions(diffusion_projection_unicycle_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    if (N > 0)
    {
        // constraints.constr_type == "BGH" and dims.nh > 0
        capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun_jac[i], diffusion_projection_unicycle_constr_h_fun_jac_uxt_zt);
        }
        capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
        for (int i = 0; i < N-1; i++) {
            MAP_CASADI_FNC(nl_constr_h_fun[i], diffusion_projection_unicycle_constr_h_fun);
        }
    



    
        // explicit ode
        capsule->expl_vde_forw = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_forw[i], diffusion_projection_unicycle_expl_vde_forw);
        }

        

        capsule->expl_ode_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_ode_fun[i], diffusion_projection_unicycle_expl_ode_fun);
        }

        capsule->expl_vde_adj = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
        for (int i = 0; i < N; i++) {
            MAP_CASADI_FNC(expl_vde_adj[i], diffusion_projection_unicycle_expl_vde_adj);
        }

    
    } // N > 0
    MAP_CASADI_FNC(nl_constr_h_e_fun_jac, diffusion_projection_unicycle_constr_h_e_fun_jac_uxt_zt);
    MAP_CASADI_FNC(nl_constr_h_e_fun, diffusion_projection_unicycle_constr_h_e_fun);
    
    

#undef MAP_CASADI_FNC
}


/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 5
 */
void diffusion_projection_unicycle_acados_create_set_default_parameters(diffusion_projection_unicycle_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        diffusion_projection_unicycle_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 5
 */
void diffusion_projection_unicycle_acados_setup_nlp_in(diffusion_projection_unicycle_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        diffusion_projection_unicycle_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.05;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.05;
        cost_scaling[1] = 0.05;
        cost_scaling[2] = 0.05;
        cost_scaling[3] = 0.05;
        cost_scaling[4] = 0.05;
        cost_scaling[5] = 0.05;
        cost_scaling[6] = 0.05;
        cost_scaling[7] = 0.05;
        cost_scaling[8] = 0.05;
        cost_scaling[9] = 0.05;
        cost_scaling[10] = 0.05;
        cost_scaling[11] = 0.05;
        cost_scaling[12] = 0.05;
        cost_scaling[13] = 0.05;
        cost_scaling[14] = 0.05;
        cost_scaling[15] = 0.05;
        cost_scaling[16] = 0.05;
        cost_scaling[17] = 0.05;
        cost_scaling[18] = 0.05;
        cost_scaling[19] = 0.05;
        cost_scaling[20] = 0.05;
        cost_scaling[21] = 0.05;
        cost_scaling[22] = 0.05;
        cost_scaling[23] = 0.05;
        cost_scaling[24] = 0.05;
        cost_scaling[25] = 0.05;
        cost_scaling[26] = 0.05;
        cost_scaling[27] = 0.05;
        cost_scaling[28] = 0.05;
        cost_scaling[29] = 0.05;
        cost_scaling[30] = 0.05;
        cost_scaling[31] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->expl_vde_forw[i]);
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "expl_vde_adj", &capsule->expl_vde_adj[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 100;
    W_0[1+(NY0) * 1] = 100;
    W_0[2+(NY0) * 2] = 0.01;
    W_0[3+(NY0) * 3] = 0.01;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* Vx_0 = calloc(NY0*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_0[0+(NY0) * 0] = 1;
    Vx_0[1+(NY0) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vx", Vx_0);
    free(Vx_0);
    double* Vu_0 = calloc(NY0*NU, sizeof(double));
    // change only the non-zero elements:
    Vu_0[2+(NY0) * 0] = 1;
    Vu_0[3+(NY0) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "Vu", Vu_0);
    free(Vu_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 100;
    W[1+(NY) * 1] = 100;
    W[2+(NY) * 2] = 0.01;
    W[3+(NY) * 3] = 0.01;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* Vx = calloc(NY*NX, sizeof(double));
    // change only the non-zero elements:
    Vx[0+(NY) * 0] = 1;
    Vx[1+(NY) * 1] = 1;
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vx", Vx);
    }
    free(Vx);

    
    double* Vu = calloc(NY*NU, sizeof(double));
    // change only the non-zero elements:
    Vu[2+(NY) * 0] = 1;
    Vu[3+(NY) * 1] = 1;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Vu", Vu);
    }
    free(Vu);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 1;
    W_e[1+(NYN) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    double* Vx_e = calloc(NYN*NX, sizeof(double));
    // change only the non-zero elements:
    Vx_e[0+(NYN) * 0] = 1;
    Vx_e[1+(NYN) * 1] = 1;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Vx", Vx_e);
    free(Vx_e);




    // slacks
    double* zlumem = calloc(4*NS, sizeof(double));
    double* Zl = zlumem+NS*0;
    double* Zu = zlumem+NS*1;
    double* zl = zlumem+NS*2;
    double* zu = zlumem+NS*3;
    // change only the non-zero elements:
    Zl[0] = 10;
    Zl[1] = 10;
    Zl[2] = 10;
    Zl[3] = 10;
    Zl[4] = 10;
    Zl[5] = 10;
    Zl[6] = 10;
    Zl[7] = 10;
    Zl[8] = 10;
    Zl[9] = 10;
    Zl[10] = 10;
    Zl[11] = 10;
    Zl[12] = 10;
    Zl[13] = 10;
    Zl[14] = 10;
    Zl[15] = 10;
    Zl[16] = 10;
    Zl[17] = 10;
    Zl[18] = 10;
    Zl[19] = 10;
    Zl[20] = 10;
    Zl[21] = 10;
    Zl[22] = 10;
    Zl[23] = 10;
    Zl[24] = 10;
    Zl[25] = 10;
    Zl[26] = 10;
    Zl[27] = 10;
    Zl[28] = 10;
    Zl[29] = 10;
    Zu[0] = 10;
    Zu[1] = 10;
    Zu[2] = 10;
    Zu[3] = 10;
    Zu[4] = 10;
    Zu[5] = 10;
    Zu[6] = 10;
    Zu[7] = 10;
    Zu[8] = 10;
    Zu[9] = 10;
    Zu[10] = 10;
    Zu[11] = 10;
    Zu[12] = 10;
    Zu[13] = 10;
    Zu[14] = 10;
    Zu[15] = 10;
    Zu[16] = 10;
    Zu[17] = 10;
    Zu[18] = 10;
    Zu[19] = 10;
    Zu[20] = 10;
    Zu[21] = 10;
    Zu[22] = 10;
    Zu[23] = 10;
    Zu[24] = 10;
    Zu[25] = 10;
    Zu[26] = 10;
    Zu[27] = 10;
    Zu[28] = 10;
    Zu[29] = 10;
    zl[0] = 10000;
    zl[1] = 10000;
    zl[2] = 10000;
    zl[3] = 10000;
    zl[4] = 10000;
    zl[5] = 10000;
    zl[6] = 10000;
    zl[7] = 10000;
    zl[8] = 10000;
    zl[9] = 10000;
    zl[10] = 10000;
    zl[11] = 10000;
    zl[12] = 10000;
    zl[13] = 10000;
    zl[14] = 10000;
    zl[15] = 10000;
    zl[16] = 10000;
    zl[17] = 10000;
    zl[18] = 10000;
    zl[19] = 10000;
    zl[20] = 10000;
    zl[21] = 10000;
    zl[22] = 10000;
    zl[23] = 10000;
    zl[24] = 10000;
    zl[25] = 10000;
    zl[26] = 10000;
    zl[27] = 10000;
    zl[28] = 10000;
    zl[29] = 10000;
    zu[0] = 10000;
    zu[1] = 10000;
    zu[2] = 10000;
    zu[3] = 10000;
    zu[4] = 10000;
    zu[5] = 10000;
    zu[6] = 10000;
    zu[7] = 10000;
    zu[8] = 10000;
    zu[9] = 10000;
    zu[10] = 10000;
    zu[11] = 10000;
    zu[12] = 10000;
    zu[13] = 10000;
    zu[14] = 10000;
    zu[15] = 10000;
    zu[16] = 10000;
    zu[17] = 10000;
    zu[18] = 10000;
    zu[19] = 10000;
    zu[20] = 10000;
    zu[21] = 10000;
    zu[22] = 10000;
    zu[23] = 10000;
    zu[24] = 10000;
    zu[25] = 10000;
    zu[26] = 10000;
    zu[27] = 10000;
    zu[28] = 10000;
    zu[29] = 10000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zl", Zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "Zu", Zu);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zl", zl);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "zu", zu);
    }
    free(zlumem);


    // slacks terminal
    double* zluemem = calloc(4*NSN, sizeof(double));
    double* Zl_e = zluemem+NSN*0;
    double* Zu_e = zluemem+NSN*1;
    double* zl_e = zluemem+NSN*2;
    double* zu_e = zluemem+NSN*3;

    // change only the non-zero elements:
    Zl_e[0] = 10;
    Zl_e[1] = 10;
    Zl_e[2] = 10;
    Zl_e[3] = 10;
    Zl_e[4] = 10;
    Zl_e[5] = 10;
    Zl_e[6] = 10;
    Zl_e[7] = 10;
    Zl_e[8] = 10;
    Zl_e[9] = 10;
    Zl_e[10] = 10;
    Zl_e[11] = 10;
    Zl_e[12] = 10;
    Zl_e[13] = 10;
    Zl_e[14] = 10;
    Zl_e[15] = 10;
    Zl_e[16] = 10;
    Zl_e[17] = 10;
    Zl_e[18] = 10;
    Zl_e[19] = 10;
    Zl_e[20] = 10;
    Zl_e[21] = 10;
    Zl_e[22] = 10;
    Zl_e[23] = 10;
    Zl_e[24] = 10;
    Zl_e[25] = 10;
    Zl_e[26] = 10;
    Zl_e[27] = 10;
    Zl_e[28] = 10;
    Zl_e[29] = 10;
    Zu_e[0] = 10;
    Zu_e[1] = 10;
    Zu_e[2] = 10;
    Zu_e[3] = 10;
    Zu_e[4] = 10;
    Zu_e[5] = 10;
    Zu_e[6] = 10;
    Zu_e[7] = 10;
    Zu_e[8] = 10;
    Zu_e[9] = 10;
    Zu_e[10] = 10;
    Zu_e[11] = 10;
    Zu_e[12] = 10;
    Zu_e[13] = 10;
    Zu_e[14] = 10;
    Zu_e[15] = 10;
    Zu_e[16] = 10;
    Zu_e[17] = 10;
    Zu_e[18] = 10;
    Zu_e[19] = 10;
    Zu_e[20] = 10;
    Zu_e[21] = 10;
    Zu_e[22] = 10;
    Zu_e[23] = 10;
    Zu_e[24] = 10;
    Zu_e[25] = 10;
    Zu_e[26] = 10;
    Zu_e[27] = 10;
    Zu_e[28] = 10;
    Zu_e[29] = 10;
    zl_e[0] = 10000;
    zl_e[1] = 10000;
    zl_e[2] = 10000;
    zl_e[3] = 10000;
    zl_e[4] = 10000;
    zl_e[5] = 10000;
    zl_e[6] = 10000;
    zl_e[7] = 10000;
    zl_e[8] = 10000;
    zl_e[9] = 10000;
    zl_e[10] = 10000;
    zl_e[11] = 10000;
    zl_e[12] = 10000;
    zl_e[13] = 10000;
    zl_e[14] = 10000;
    zl_e[15] = 10000;
    zl_e[16] = 10000;
    zl_e[17] = 10000;
    zl_e[18] = 10000;
    zl_e[19] = 10000;
    zl_e[20] = 10000;
    zl_e[21] = 10000;
    zl_e[22] = 10000;
    zl_e[23] = 10000;
    zl_e[24] = 10000;
    zl_e[25] = 10000;
    zl_e[26] = 10000;
    zl_e[27] = 10000;
    zl_e[28] = 10000;
    zl_e[29] = 10000;
    zu_e[0] = 10000;
    zu_e[1] = 10000;
    zu_e[2] = 10000;
    zu_e[3] = 10000;
    zu_e[4] = 10000;
    zu_e[5] = 10000;
    zu_e[6] = 10000;
    zu_e[7] = 10000;
    zu_e[8] = 10000;
    zu_e[9] = 10000;
    zu_e[10] = 10000;
    zu_e[11] = 10000;
    zu_e[12] = 10000;
    zu_e[13] = 10000;
    zu_e[14] = 10000;
    zu_e[15] = 10000;
    zu_e[16] = 10000;
    zu_e[17] = 10000;
    zu_e[18] = 10000;
    zu_e[19] = 10000;
    zu_e[20] = 10000;
    zu_e[21] = 10000;
    zu_e[22] = 10000;
    zu_e[23] = 10000;
    zu_e[24] = 10000;
    zu_e[25] = 10000;
    zu_e[26] = 10000;
    zu_e[27] = 10000;
    zu_e[28] = 10000;
    zu_e[29] = 10000;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);
    free(zluemem);

    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(5 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);












    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -1.5;
    ubu[0] = 1.5;
    lbu[1] = -3.141592653589793;
    ubu[1] = 3.141592653589793;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);






    /* Path constraints */

    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 3;
    idxbx[1] = 4;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    ubx[0] = 1;
    lbx[1] = -3.141592653589793;
    ubx[1] = 3.141592653589793;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);


    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    uh[0] = 1000000;
    uh[1] = 1000000;
    uh[2] = 1000000;
    uh[3] = 1000000;
    uh[4] = 1000000;
    uh[5] = 1000000;
    uh[6] = 1000000;
    uh[7] = 1000000;
    uh[8] = 1000000;
    uh[9] = 1000000;
    uh[10] = 1000000;
    uh[11] = 1000000;
    uh[12] = 1000000;
    uh[13] = 1000000;
    uh[14] = 1000000;
    uh[15] = 1000000;
    uh[16] = 1000000;
    uh[17] = 1000000;
    uh[18] = 1000000;
    uh[19] = 1000000;
    uh[20] = 1000000;
    uh[21] = 1000000;
    uh[22] = 1000000;
    uh[23] = 1000000;
    uh[24] = 1000000;
    uh[25] = 1000000;
    uh[26] = 1000000;
    uh[27] = 1000000;
    uh[28] = 1000000;
    uh[29] = 1000000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "uh", uh);
        
        
    }
    free(luh);








    // set up soft bounds for nonlinear constraints
    int* idxsh = malloc(NSH * sizeof(int));
    idxsh[0] = 0;
    idxsh[1] = 1;
    idxsh[2] = 2;
    idxsh[3] = 3;
    idxsh[4] = 4;
    idxsh[5] = 5;
    idxsh[6] = 6;
    idxsh[7] = 7;
    idxsh[8] = 8;
    idxsh[9] = 9;
    idxsh[10] = 10;
    idxsh[11] = 11;
    idxsh[12] = 12;
    idxsh[13] = 13;
    idxsh[14] = 14;
    idxsh[15] = 15;
    idxsh[16] = 16;
    idxsh[17] = 17;
    idxsh[18] = 18;
    idxsh[19] = 19;
    idxsh[20] = 20;
    idxsh[21] = 21;
    idxsh[22] = 22;
    idxsh[23] = 23;
    idxsh[24] = 24;
    idxsh[25] = 25;
    idxsh[26] = 26;
    idxsh[27] = 27;
    idxsh[28] = 28;
    idxsh[29] = 29;
    double* lush = calloc(2*NSH, sizeof(double));
    double* lsh = lush;
    double* ush = lush + NSH;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxsh", idxsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lsh", lsh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ush", ush);
    }
    free(idxsh);
    free(lush);



    /* terminal constraints */





    // set up nonlinear constraints for last stage
    double* luh_e = calloc(2*NHN, sizeof(double));
    double* lh_e = luh_e;
    double* uh_e = luh_e + NHN;
    uh_e[0] = 1000000;
    uh_e[1] = 1000000;
    uh_e[2] = 1000000;
    uh_e[3] = 1000000;
    uh_e[4] = 1000000;
    uh_e[5] = 1000000;
    uh_e[6] = 1000000;
    uh_e[7] = 1000000;
    uh_e[8] = 1000000;
    uh_e[9] = 1000000;
    uh_e[10] = 1000000;
    uh_e[11] = 1000000;
    uh_e[12] = 1000000;
    uh_e[13] = 1000000;
    uh_e[14] = 1000000;
    uh_e[15] = 1000000;
    uh_e[16] = 1000000;
    uh_e[17] = 1000000;
    uh_e[18] = 1000000;
    uh_e[19] = 1000000;
    uh_e[20] = 1000000;
    uh_e[21] = 1000000;
    uh_e[22] = 1000000;
    uh_e[23] = 1000000;
    uh_e[24] = 1000000;
    uh_e[25] = 1000000;
    uh_e[26] = 1000000;
    uh_e[27] = 1000000;
    uh_e[28] = 1000000;
    uh_e[29] = 1000000;

    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "uh", uh_e);
    
    
    free(luh_e);



    /* terminal soft constraints */








    // set up soft bounds for nonlinear constraints
    int* idxsh_e = malloc(NSHN * sizeof(int));
    idxsh_e[0] = 0;
    idxsh_e[1] = 1;
    idxsh_e[2] = 2;
    idxsh_e[3] = 3;
    idxsh_e[4] = 4;
    idxsh_e[5] = 5;
    idxsh_e[6] = 6;
    idxsh_e[7] = 7;
    idxsh_e[8] = 8;
    idxsh_e[9] = 9;
    idxsh_e[10] = 10;
    idxsh_e[11] = 11;
    idxsh_e[12] = 12;
    idxsh_e[13] = 13;
    idxsh_e[14] = 14;
    idxsh_e[15] = 15;
    idxsh_e[16] = 16;
    idxsh_e[17] = 17;
    idxsh_e[18] = 18;
    idxsh_e[19] = 19;
    idxsh_e[20] = 20;
    idxsh_e[21] = 21;
    idxsh_e[22] = 22;
    idxsh_e[23] = 23;
    idxsh_e[24] = 24;
    idxsh_e[25] = 25;
    idxsh_e[26] = 26;
    idxsh_e[27] = 27;
    idxsh_e[28] = 28;
    idxsh_e[29] = 29;
    double* lush_e = calloc(2*NSHN, sizeof(double));
    double* lsh_e = lush_e;
    double* ush_e = lush_e + NSHN;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "idxsh", idxsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "lsh", lsh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, N, "ush", ush_e);
    free(idxsh_e);
    free(lush_e);





}


static void diffusion_projection_unicycle_acados_create_set_opts(diffusion_projection_unicycle_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 2;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 31;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    bool with_anderson_acceleration = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_anderson_acceleration", &with_anderson_acceleration);

    double anderson_activation_threshold = 10;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "anderson_activation_threshold", &anderson_activation_threshold);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 7
 */
void diffusion_projection_unicycle_acados_set_nlp_out(diffusion_projection_unicycle_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for diffusion_projection_unicycle_acados_create: step 9
 */
int diffusion_projection_unicycle_acados_create_precompute(diffusion_projection_unicycle_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int diffusion_projection_unicycle_acados_create_with_discretization(diffusion_projection_unicycle_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != DIFFUSION_PROJECTION_UNICYCLE_N && !new_time_steps) {
        fprintf(stderr, "diffusion_projection_unicycle_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, DIFFUSION_PROJECTION_UNICYCLE_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    diffusion_projection_unicycle_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = diffusion_projection_unicycle_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    diffusion_projection_unicycle_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    diffusion_projection_unicycle_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    diffusion_projection_unicycle_acados_create_setup_functions(capsule);
    diffusion_projection_unicycle_acados_setup_nlp_in(capsule, N, new_time_steps);
    diffusion_projection_unicycle_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = diffusion_projection_unicycle_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int diffusion_projection_unicycle_acados_update_qp_solver_cond_N(diffusion_projection_unicycle_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from diffusion_projection_unicycle_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = diffusion_projection_unicycle_acados_create_precompute(capsule);
    return status;
}


int diffusion_projection_unicycle_acados_reset(diffusion_projection_unicycle_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int diffusion_projection_unicycle_acados_update_params(diffusion_projection_unicycle_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 90;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int diffusion_projection_unicycle_acados_update_params_sparse(diffusion_projection_unicycle_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int diffusion_projection_unicycle_acados_set_p_global_and_precompute_dependencies(diffusion_projection_unicycle_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, diffusion_projection_unicycle_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int diffusion_projection_unicycle_acados_solve(diffusion_projection_unicycle_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int diffusion_projection_unicycle_acados_setup_qp_matrices_and_factorize(diffusion_projection_unicycle_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}






int diffusion_projection_unicycle_acados_free(diffusion_projection_unicycle_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->expl_vde_forw[i]);
        
        external_function_external_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_external_param_casadi_free(&capsule->expl_vde_adj[i]);
    }
    free(capsule->expl_vde_adj);
    free(capsule->expl_vde_forw);
    
    free(capsule->expl_ode_fun);

    // cost

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_external_param_casadi_free(&capsule->nl_constr_h_e_fun);



    return 0;
}


void diffusion_projection_unicycle_acados_print_stats(diffusion_projection_unicycle_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    int stat_n_max = 16;
    if (stat_n > stat_n_max)
    {
        printf("stat_n_max = %d is too small, increase it in the template!\n", stat_n_max);
        exit(1);
    }
    double stat[1616];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int diffusion_projection_unicycle_acados_custom_update(diffusion_projection_unicycle_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *diffusion_projection_unicycle_acados_get_nlp_in(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *diffusion_projection_unicycle_acados_get_nlp_out(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *diffusion_projection_unicycle_acados_get_sens_out(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *diffusion_projection_unicycle_acados_get_nlp_solver(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *diffusion_projection_unicycle_acados_get_nlp_config(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_config; }
void *diffusion_projection_unicycle_acados_get_nlp_opts(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *diffusion_projection_unicycle_acados_get_nlp_dims(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *diffusion_projection_unicycle_acados_get_nlp_plan(diffusion_projection_unicycle_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
