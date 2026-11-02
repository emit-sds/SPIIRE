#! /usr/bin/env python
#
#  Copyright 2026 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors: Jay E. Fahlen, jay.e.fahlen@gjpl.nasa.gov

import numpy as np
import pdb
import jax.numpy as jnp

import plume_models

def objective(x_params_all, mf_data, unc, sens, good_pixel_flags, 
              nx, ny, dx, dy, x0_grid, y0_grid, pnt_list,
              plume_x_m_list, plume_y_m_list,
              plume_mean_locations_spacing_m,
              theta_deg_list,
              do_exp):

    n_params = len(x_params_all)

    mf_model = forward_list(x_params_all, nx, ny, dx, dy, x0_grid, y0_grid, pnt_list,
                            plume_x_m_list, plume_y_m_list,
                            plume_mean_locations_spacing_m,
                            theta_deg_list,
                            do_exp)

    # Cost with sensitivity and uncertainty
    #cost = (sens * mf_model - mf_data)**2 / (unc * sens)**2
    us = unc * sens
    us_no0 = jnp.where(us == 0, 1., us)
    cost = jnp.where(us == 0., 0., (sens * mf_model - mf_data)**2 / (us_no0)**2)
    cost = jnp.where(jnp.isfinite(cost) == True, cost, 0)
    for pnt in pnt_list:
        cost.at[pnt].set(0.)
    cost = 0.5 * jnp.nansum(jnp.where(good_pixel_flags == True, cost, 0))

    return cost

def forward_list(x_list, 
            nx, ny, dx, dy, x0_grid, y0_grid, pnt_list,
            plume_x_m_list, plume_y_m_list,
            plume_mean_locations_spacing_m,
            theta_deg_list,
            do_exp):

    n_params = len(x_list)
    n_params_per = n_params // len(pnt_list)
    mf_models = []
    for plume_num, (pnt, plume_x_m, plume_y_m, theta_deg) in \
                    enumerate(zip(pnt_list, plume_x_m_list, plume_y_m_list, theta_deg_list)):
        x = x_list[plume_num*n_params_per:(plume_num+1)*n_params_per]
        f = forward(x, 
                    nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                    plume_x_m, plume_y_m,
                    plume_mean_locations_spacing_m,
                    theta_deg, do_exp)
        mf_models.append(f)
    f = jnp.sum(jnp.array(mf_models), axis = 0)
    
    return f

def forward(x, 
            nx, ny, dx, dy, x0_grid, y0_grid, pnt,
            plume_x_m, plume_y_m,
            plume_mean_locations_spacing_m,
            theta_deg,
            do_exp):
    
    C_Q_kg_per_s_mpers = x[0]
    a = x[1]
    exponent = x[2]
    x0 = 1000.

    plume_mean_locations = None
    if theta_deg is None:
        theta_deg = x[3]
        if len(x) > 4:
            plume_mean_locations = x[4:]
    else:
        if len(x) > 3:
            plume_mean_locations = x[3:]

    model = plume_models.point_gaussian_plume_model(nx, ny, dx, dy, x0_grid, y0_grid, '',
                                        plume_x_m, plume_y_m,
                                        plume_mean_locations_spacing_m,
                                        C_Q_kg_per_s_mpers, x0, a, exponent, theta_deg, plume_mean_locations,
                                        do_exp)
    return model



def objective_derivs(x, mf_data, unc, sens, good_pixel_flags, base_plume_parameters, pm, retrieval_params, return_derivatives = True,
                     d_1stmom_m = None, d_2ndmom_m = None, return_cost_image = False, use_width_based_cost_weighting = False):

    if return_derivatives:
        mf_model, mf_model_derivs_dict = forward(x, base_plume_parameters, pm, retrieval_params, do_derivs = True, d_1stmom_m = d_1stmom_m)
    else:
        mf_model = forward(x, base_plume_parameters, pm, retrieval_params, do_derivs = False, d_1stmom_m = d_1stmom_m)

    if use_width_based_cost_weighting:
        if d_2ndmom_m is None:
            raise ValueError(f'Cannot use width-based cost_weighting when d_2ndmom_m is None')

    d_2ndmom_rotated_m = 1.
    if d_2ndmom_m is not None:
        d_2ndmom_rotated_m = pm.get_rotated_function(np.abs(d_2ndmom_m))

    finite = np.logical_not(np.isfinite(mf_model))
    
    # Cost with sensitivity and uncertainty
    cost = (sens * mf_model - mf_data)**2 / (unc * sens)**2

    if use_width_based_cost_weighting:
        cost = cost / d_2ndmom_rotated_m
        cost[d_2ndmom_rotated_m < 0.01] = 0

    cost[finite] = 0

    #ignore = mf_data < 0
    #cost[ignore] = 0

    if return_cost_image:
        return 0.5*cost, 1 / d_2ndmom_rotated_m

    cost = 0.5 * np.nansum(cost[good_pixel_flags])

    if return_derivatives:
        cost_derivs = []
        cd_fac = (sens * mf_model - mf_data) / unc**2 / sens
        if use_width_based_cost_weighting:
            cd_fac = cd_fac / d_2ndmom_rotated_m

        for retrieval_param in retrieval_params:
            cd = cd_fac * mf_model_derivs_dict[retrieval_param]
            cd[finite] = 0
            #cd[ignore] = 0
            cd = np.nansum(cd[good_pixel_flags])
            cost_derivs.append(cd)

    if return_derivatives:
        cost_derivs = np.array(cost_derivs)
        return cost, cost_derivs
    else:
        return cost

def forward_derivs(x, base_plume_parameters, pm, retrieval_params, do_derivs = False, do_full_scene = False, d_1stmom_m = None, d_2ndmom_m = None):
    
    plume_parameters_dict = {k:x for k,x in base_plume_parameters.items()}
    for retrieval_param, xval in zip(retrieval_params, x):
        plume_parameters_dict[retrieval_param] = xval
    
    if d_1stmom_m is None:
        if len(x) > len(retrieval_params):
            plume_mean_locations = x[len(retrieval_params):]
            #plume_mean_locations = scipy.signal.savgol_filter(plume_mean_locations, 19, 2)
            #plume_mean_locations = scipy.signal.savgol_filter(plume_mean_locations, 5, 2)
            #plume_mean_locations = scipy.signal.savgol_filter(plume_mean_locations, 11, 2)
            #plume_mean_locations = scipy.signal.savgol_filter(plume_mean_locations, 3, 2)
            #plume_mean_locations = np.polyval(plume_mean_locations, np.arange(len(d_1stmom_m)))
        else:
            plume_mean_locations = None
    else:
        plume_mean_locations = d_1stmom_m
        if len(x) > len(retrieval_params):
            raise ValueError('Cannot have extra retrieval_params if d_1stmom_m is not None')

    if do_derivs:
        model = pm.get_plume_empirical_with_derivatives(local_plume_parameters_dict = plume_parameters_dict, do_full_scene = do_full_scene, 
                                                        plume_mean_locations = plume_mean_locations)
    else:
        model = pm.get_plume_empirical_no_derivatives(local_plume_parameters_dict = plume_parameters_dict, do_full_scene = do_full_scene, 
                                                      plume_mean_locations = plume_mean_locations)
    return model

