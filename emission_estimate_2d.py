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

import argparse
import os
import numpy as np
import pandas as pd
from time import time
import pdb
import scipy.optimize
import scipy.stats
import scipy.spatial
import pickle
import itertools
import geopandas as gpd
import yaml
import shutil
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
#jax.config. update("jax_debug_nans", True)
import jaxopt

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
plt.ioff()
from matplotlib import patches, path, transforms
from mpl_toolkits.axes_grid1 import ImageGrid

import plume_models
import load_data
from compute_cross_plume_slices import compute_cross_plume_slices
from objective_functions_for_2d_inversion import objective, forward_list

def run_one(cfg, plume_ids,
            plot_cost = False, 
            save_all_dicts = False, 
            save_data_dict = False):
    '''Run the full staged optimization for one plume'''

    print(plume_ids)
    n_plumes = len(plume_ids)

    if 'output_name_tag' in list(cfg.keys()):
        cfg['output_name_tag'] = '_' + cfg['output_name_tag']
    else:
        cfg['output_name_tag'] = ''

    instrument = cfg['instrument'] if 'instrument' in list(cfg.keys()) else None
    input_data_location = cfg['input_data_location'] if 'input_data_location' in list(cfg.keys()) else None
    #try:
    data_dicts = []
    for plume_id in plume_ids:
        data_dict = load_data.load_MF_from_plume_ID(plume_id,
                                                    plume_json_filename = cfg['plume_json_filename'],
                                                    show_plots = False,
                                                    instrument = instrument,
                                                    input_data_location = input_data_location)
        data_dicts.append(data_dict)
    

    if 'sim_plume_parameters' in list(cfg.keys()):
        sim_dicts = []
        for plume_num, sp, in enumerate(cfg['sim_plume_parameters']):
            #sp = cfg['sim_plume_parameters']
            data_dict = data_dicts[plume_num]

            xoff = sp['plume_x_m'] - data_dict['pseudo_origin_ort_x']
            yoff = sp['plume_y_m'] - data_dict['pseudo_origin_ort_y']

            data_dict['pseudo_origin_ort_x'] = data_dict['pseudo_origin_ort_x'] + xoff
            data_dict['pseudo_origin_ort_y'] = data_dict['pseudo_origin_ort_y'] + yoff

            data_dict['plume_angle_ccw_from_x_ort_rad'] = sp['plume_angle_ccw_from_x_ort_deg'] * np.pi / 180

            dx = data_dict['gsd_ort_m']
            dy = dx # Assume they're all square
            init_plume_parameters = {k:d for k, d in cfg['init_plume_parameters'].items()}

            init_plume_parameters['plume_x_m'] = data_dict['pseudo_origin_ort_x'] * dx
            init_plume_parameters['plume_y_m'] = data_dict['pseudo_origin_ort_y'] * dy
            init_plume_parameters['theta_deg'] = sp['theta_deg']

            init_plume_parameters['C_Q_kg_per_s_mpers'] = np.log(sp['C_Q_kg_per_s_mpers'])
            init_plume_parameters['a'] = np.log(sp['a'])
            init_plume_parameters['exponent'] = sp['exponent']
            init_plume_parameters['wind_speed_m_per_s'] = 1. # Required for plume model code, but should not be change from 1.

            nx, ny = data_dict['concentration_length_ort_ppmm'].shape
            x0 = np.arange(nx)[:,np.newaxis] * np.ones((nx, ny)) * dx
            y0 = (np.arange(ny)[:,np.newaxis] * np.ones((ny, nx))).T * dy

            off = 10
            xs = slice(data_dict['min_mask_ort_x']-off, data_dict['max_mask_ort_x']+off)
            ys = slice(data_dict['min_mask_ort_y']-off, data_dict['max_mask_ort_y']+off)
            retrieval_mask = np.zeros((nx, ny), dtype = bool)
            retrieval_mask[xs, ys] = True

            x0_grid = x0 #[retrieval_mask]
            y0_grid = y0 #[retrieval_mask]
            px = np.round(init_plume_parameters['plume_x_m']/dx).astype(int)
            py = np.round(init_plume_parameters['plume_y_m']/dy).astype(int)
            tmp = np.zeros(retrieval_mask.shape, dtype = bool)
            tmp[px,py] = True

            #pm = plume_models.PlumeModel(nx, dx, ny, dy, init_plume_parameters)
            #model = pm.get_plume_empirical_no_derivatives(local_plume_parameters_dict = init_plume_parameters)
            model = plume_models.point_gaussian_plume_model(nx, ny, dx, dy, x0_grid, y0_grid, '',
                                                            init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
                                                            0.,
                                                            init_plume_parameters['C_Q_kg_per_s_mpers'], 1000.,
                                                            init_plume_parameters['a'], init_plume_parameters['exponent'],
                                                            init_plume_parameters['theta_deg'], None, True)

            pnt = np.unravel_index(np.argmax(model), model.shape)
            model = np.array(model)
            model[pnt] = 0.
            data_dict['concentration_length_ort_ppmm'] = 0.*data_dict['concentration_length_ort_ppmm'] + model * data_dict['sensitivity']

            pct = 0.1
            inds = np.array(np.where(model > pct*np.nanmax(model))).T
            hull = scipy.spatial.ConvexHull(inds)
            hull_inds = inds[hull.vertices,:]

            data_dict['min_mask_ort_x'], data_dict['min_mask_ort_y'] = np.min(hull_inds, axis = 0)
            data_dict['max_mask_ort_x'], data_dict['max_mask_ort_y'] = np.max(hull_inds, axis = 0)

            data_dict['plume_ort_x'], data_dict['plume_ort_y'] = hull_inds.T
            sim_dicts.append(data_dict)
        
        data_dicts[0]['concentration_length_ort_ppmm'] = np.sum([dd['concentration_length_ort_ppmm'] for dd in data_dicts], axis = 0)
        
    data_dict = {
        'plume_ids': [d['plume_id'] for d in data_dicts],
        'FID': data_dicts[0]['FID'],
        'concentration_length_ort_ppmm': data_dicts[0]['concentration_length_ort_ppmm'],
        'sensitivity': data_dicts[0]['sensitivity'],
        'uncertainty': data_dicts[0]['uncertainty'],
        'plume_ort_x_list': [d['plume_ort_x'] for d in data_dicts],
        'plume_ort_y_list': [d['plume_ort_y'] for d in data_dicts],
        'pseudo_origin_ort_x_list': [d['pseudo_origin_ort_x'] for d in data_dicts],
        'pseudo_origin_ort_y_list': [d['pseudo_origin_ort_y'] for d in data_dicts],
        'center_of_mass_ort_x_list': [d['center_of_mass_ort_x'] for d in data_dicts],
        'center_of_mass_ort_y_list': [d['center_of_mass_ort_y'] for d in data_dicts],
        'min_mask_ort_x': np.min([d['min_mask_ort_x'] for d in data_dicts]),
        'min_mask_ort_y': np.min([d['min_mask_ort_y'] for d in data_dicts]),
        'max_mask_ort_x': np.max([d['max_mask_ort_x'] for d in data_dicts]),
        'max_mask_ort_y': np.max([d['max_mask_ort_y'] for d in data_dicts]),
        'plume_mask_ort_list': [d['plume_mask_ort'] for d in data_dicts],
        'plume_angle_ccw_from_x_ort_rad_list': [d['plume_angle_ccw_from_x_ort_rad'] for d in data_dicts],
        'gsd_ort_m': data_dicts[0]['gsd_ort_m'],
        'wind_speed_m_per_s': None,
        'instrument_name': data_dicts[0]['instrument_name'],
        'good_pixels_list': [d['good_pixels'] for d in data_dicts], 
        'scene_date': data_dicts[0]['scene_date'],
        'plume_time_UTC': data_dicts[0]['plume_time_UTC']}
    
    plume_ids_str = '_'.join(plume_ids)

    load_success = True
    #except:
    #    load_success = False
    #    r = {'plume_id': plume_id, 'success': 'load data failed'}
    #    r_new = {'plume_id': plume_id, 'success': 'load data failed'}

    if load_success:
        #try:
        # First stage - grid search for the plume origin and the wind direction
        cost = np.zeros((cfg['nxb'], cfg['nyb']))
        res = []
        xr, yr = (np.arange(cfg['nxb']) - cfg['nxb']//2)*cfg['skip'], (np.arange(cfg['nyb']) - cfg['nyb']//2)*cfg['skip']
        for jb, yb in enumerate(yr):
            resx = []
            for ib, xb in enumerate(xr):
                outname = f'{cfg["output_path"]}/{plume_ids_str}_{ib}_{jb}'
                r = estimate_plume(data_dict, cfg, 0, 0.,
                                    pseudo_origin_x_offset_bin_units = xb,
                                    pseudo_origin_y_offset_bin_units = yb,
                                    show_plots = True, close_plot = True)
                if save_all_dicts:
                    pickle.dump(r, open(outname + cfg['output_name_tag'] + '.pickle', 'wb'))
                r['xoff'] = xb
                r['yoff'] = yb
                cost[ib,jb] = r['fun']
                resx.append(r)
            res.append(resx)
        mx, my = np.unravel_index(np.argmin(cost), cost.shape)
        r = res[my][mx] # The list ordering makes x and y reversed here
        r['outname'] = f'{cfg["output_path"]}/{plume_ids_str}{cfg["output_name_tag"]}'
        #if 'sim_plume_parameters' in list(cfg.keys()):
        #    r['sim_C_Q_kg_per_s_mpers'] = sim_plume_parameters['C_Q_kg_per_s_mpers']
        #    r['sim_a'] = sim_plume_parameters['a']
        #    r['sim_exponent'] = sim_plume_parameters['exponent']
        #    r['sim_theta_deg'] = sim_plume_parameters['theta_deg']
        #    r['sim_plume_x_m'] = sim_plume_parameters['plume_x_m']
        #    r['sim_plume_y_m'] = sim_plume_parameters['plume_y_m']
        pickle.dump(r, open(f'{cfg["output_path"]}/{plume_ids_str}{cfg["output_name_tag"]}_data_dict_straight.pickle', 'wb'))

        if plot_cost:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            im = ax.imshow(cost.T, origin = 'lower', vmin = np.min(cost), vmax = np.max(cost), 
                        extent = [xr[0]-0.5, xr[-1]+0.5, yr[0]-0.5, yr[-1]+0.5])
            cbar = fig.colorbar(im, ax=ax)
            ax.set_title(f'Cost {plume_ids_str}')
            ax.set_xlabel('x shift')
            ax.set_ylabel('y shift')

        # Generate initial condition for the cross-wind mean deviation
        for plume_num in np.arange(n_plumes):
            d_1stmom_m, d_2ndmom_m, plume_length_m = compute_cross_plume_slices(r, cfg['cross_plume_n_slices'], plume_num,
                                                                                cross_length = cfg['cross_plume_cross_length'], 
                                                                                show_plots = True)
        #plt.show(block = True)

        n_bends = cfg['n_bends'] if cfg['n_bends'] > 0 else len(d_1stmom_m)
        plume_mean_locations_spacing_m = cfg['plume_mean_locations_spacing_gsd_factor'] * r['gsd_ort_m']
        if n_bends * plume_mean_locations_spacing_m > plume_length_m:
            n_bends = int(plume_length_m / plume_mean_locations_spacing_m) + 1

        print(f'\nPerforming full 2D inversion...')
        r_new = estimate_plume(data_dict, 
                                cfg,
                                n_bends, plume_mean_locations_spacing_m,
                                pseudo_origin_x_offset_bin_units = r['xoff'],
                                pseudo_origin_y_offset_bin_units = r['yoff'],
                                #d_1stmom_m = d_1stmom_m, d_2ndmom_m = d_2ndmom_m,
                                theta_deg_override = r['theta_deg_list'],
                                show_plots = True, close_plot = False)

        for plume_num in np.arange(n_plumes):
            r_new[f'theta_deg_{plume_num}'] = r[f'theta_deg_{plume_num}'] # Because theta_deg_override is provided above, this isn't retrieved
        r_new['outname'] = f'{cfg["output_path"]}/{plume_ids_str}{cfg["output_name_tag"]}'
        r_new['xoff'], r_new['yoff'] = r['xoff'], r['yoff']
        #if 'sim_plume_parameters' in list(cfg.keys()):
        #    r_new['sim_C_Q_kg_per_s_mpers'] = sim_plume_parameters['C_Q_kg_per_s_mpers']
        #    r_new['sim_a'] = sim_plume_parameters['a']
        #    r_new['sim_exponent'] = sim_plume_parameters['exponent']
        #    r_new['sim_theta_deg'] = sim_plume_parameters['theta_deg']
        #    r_new['sim_plume_x_m'] = sim_plume_parameters['plume_x_m']
        #    r_new['sim_plume_y_m'] = sim_plume_parameters['plume_y_m']
        pickle.dump(r_new, open(f'{cfg["output_path"]}/{plume_ids_str}{cfg["output_name_tag"]}_data_dict_bendy.pickle', 'wb'))

        for plume_num in np.arange(n_plumes):
            _, _, _ = compute_cross_plume_slices(r_new, n_bends, plume_num,
                                                cross_length = cfg['cross_plume_cross_length'],
                                                full_retrieval_line = r_new[f'full_retrieval_line_{plume_num}'],
                                                output_plot_basename = cfg['output_path'] + f'/{plume_ids_str}{cfg["output_name_tag"]}',
                                                show_plots = True)

        #except:
        #    r = {'plume_id': plume_id, 'success': 'estimate_plume failed'}
        #    r_new = {'plume_id': plume_id, 'success': 'estimate_plume failed'}
        
    names = ['straight', 'bendy']
    cols_no_df = ['plot_data', 'plot_model', 'plume_mask_ort_list', 'xs','ys', 'plume_ort_x_list', 
                    'plume_ort_y_list', 'uncertainty', 'sensitivity', 'cost_image', 'cost_weighting', 
                    'unc', 'sens', 'fig', 'fig_cost', 'full_retrieval_line', 'ended_near_lower_bound', 'ended_near_upper_bound',
                    'parameter_curvature', 'parameter_var', 'good_pixels_list']
    cols_no_df.extend([f'full_retrieval_line_{i}' for i in np.arange(n_plumes).astype(int)])
    # Collect and save the info
    for name, r in zip(names, [r, r_new]):
        r_for_csv = {}
        for k, d in r.items():
            if k not in cols_no_df:
                r_for_csv[k] = d
        df_results = pd.DataFrame([r_for_csv]) #.set_index('plume_id')
        #df_results = df_results[~df_results.index.duplicated(keep='first')]

        if plume_id is None:
            csv_name = cfg['output_path'] + f'/summary_{name}{cfg["output_name_tag"]}.csv'
        else:
            csv_name = cfg['output_path'] + f'/summary_{name}_{plume_ids_str}{cfg["output_name_tag"]}.csv'
        df_results.to_csv(csv_name)

    if cfg['show_plots_to_screen']:
        plt.show(block = True)

    return r, r_new

def estimate_plume(data_dict, 
                   cfg,
                   n_bends = 0,
                   plume_mean_locations_spacing_m = None,
                   pseudo_origin_x_offset_bin_units = 0, 
                   pseudo_origin_y_offset_bin_units = 0, 
                   theta_deg_override = None, 
                   show_plots = False, close_plot = True):
    '''Perform the optimization'''

    start_time = time()
    nx, ny = data_dict['concentration_length_ort_ppmm'].shape
    dx = data_dict['gsd_ort_m']
    dy = dx # Assume they're all square
    plume_ids_str = '_'.join(data_dict['plume_ids'])
    output_plot_basename = os.path.join(cfg['output_path'], plume_ids_str + cfg['output_name_tag'])

    if not os.path.exists(output_plot_basename):
        os.mkdir(output_plot_basename)
    
    if cfg['plot_iterations']:
        iter_pics = output_plot_basename + '/iter_pics/'
        if not os.path.exists(iter_pics):
            os.makedirs(iter_pics)

    retrieval_params = [x for x in cfg['retrieval_params']]
    retrieval_params = sorted(retrieval_params)

    n_plumes = len(data_dict['plume_ids'])
    init_plume_parameters_list = []
    for plume_num, _ in enumerate(data_dict['plume_ids']):
        init_plume_parameters = cfg['init_plume_parameters'].copy()
        init_plume_parameters['theta_deg'] = data_dict['plume_angle_ccw_from_x_ort_rad_list'][plume_num] * 180 / np.pi
        init_plume_parameters['plume_x_m'] = (data_dict['pseudo_origin_ort_x_list'][plume_num] + pseudo_origin_x_offset_bin_units)*dx
        init_plume_parameters['plume_y_m'] = (data_dict['pseudo_origin_ort_y_list'][plume_num] + pseudo_origin_y_offset_bin_units)*dy
        init_plume_parameters['wind_speed_m_per_s'] = 1. # Required for plume model code, but should not be change from 1.
        init_plume_parameters_list.append(init_plume_parameters)

    retrieval_bounds_list = []
    for plume_num, _ in enumerate(data_dict['plume_ids']):
        retrieval_bounds = cfg['retrieval_bounds'].copy()
        retrieval_bounds['theta_deg'] = (-15 + init_plume_parameters_list[plume_num]['theta_deg'],
                                          15 + init_plume_parameters_list[plume_num]['theta_deg'])
        retrieval_bounds_list.append(retrieval_bounds)

    theta_deg_list = [None]*n_plumes
    if theta_deg_override is not None:
        theta_deg_list = theta_deg_override
        for i, tdo in enumerate(theta_deg_override):
            init_plume_parameters_list[i]['theta_deg'] = tdo
        retrieval_params.remove('theta_deg')

    # Find retrieval region
    off = 10
    xs = slice(data_dict['min_mask_ort_x']-off, data_dict['max_mask_ort_x']+off)
    ys = slice(data_dict['min_mask_ort_y']-off, data_dict['max_mask_ort_y']+off)
    retrieval_mask = np.zeros((nx, ny), dtype = bool)
    retrieval_mask[xs, ys] = True
    square_retrieval_mask = retrieval_mask

    if cfg['use_plume_mask_or_rectangle_for_retrieval_region'] == 'plume':
        retrieval_mask = data_dict['plume_mask_ort']
    elif cfg['use_plume_mask_or_rectangle_for_retrieval_region'] == 'rectangle':
        pass # This is already what's set above
    else:
        raise ValueError(f'plume_mask_or_rectangle_for_retrieval_region {cfg["use_plume_mask_or_rectangle_for_retrieval_region"]} is not valid')
    
    # Get a plume model that is rectangular regardless of plume_mask_or_rectangle_for_retrieval_region for plotting
    x0 = np.arange(nx)[:,np.newaxis] * np.ones((nx, ny)) * dx
    y0 = (np.arange(ny)[:,np.newaxis] * np.ones((ny, nx))).T * dy
    x0_grid = x0[retrieval_mask]
    y0_grid = y0[retrieval_mask]
    pnt_list = []
    for ip in init_plume_parameters_list:
        px = np.round(ip['plume_x_m']/dx).astype(int)
        py = np.round(ip['plume_y_m']/dy).astype(int)
        tmp = np.zeros(retrieval_mask.shape, dtype = bool)
        tmp[px,py] = True
        pnt_list.append(np.where(tmp[retrieval_mask])[0])

    init_params = []
    for ip in init_plume_parameters_list:
        init_params.extend([ip[key] for key in retrieval_params] + [0.]*n_bends)

    lb, ub = [], []
    for rb in retrieval_bounds_list:
        bounds = [rb[key] for key in retrieval_params]
        lb.extend([b[0] for b in bounds])
        ub.extend([b[1] for b in bounds])
        if n_bends > 0:
            bends_bounds = cfg['cross_plume_cross_length']*data_dict['gsd_ort_m']*np.ones(n_bends)
            bends_bounds[0] = data_dict['gsd_ort_m']*1
            bends_bounds[1] = data_dict['gsd_ort_m']*2
            bends_bounds[2] = data_dict['gsd_ort_m']*3
            bends_bounds[3] = data_dict['gsd_ort_m']*4
            lb.extend(list(-1*bends_bounds))
            ub.extend(list(   bends_bounds))
    bounds = (jnp.array(lb), jnp.array(ub))

    options_key = 'trust_constr_bendy_options' if n_bends > 0 else 'trust_constr_straight_options'
    local_opt_options = cfg[options_key].copy()
    maxiter = local_opt_options['maxiter']
    del local_opt_options['maxiter']
    tol = local_opt_options['tol']
    del local_opt_options['tol']

    unc = 1 if data_dict['uncertainty'] is None else data_dict['uncertainty'][retrieval_mask]
    sens = 1 if data_dict['sensitivity'] is None else data_dict['sensitivity'][retrieval_mask]
    good_pixel_flags = np.any(data_dict['good_pixels_list'], axis=0)
    good_pixel_flags = good_pixel_flags[retrieval_mask]

    unc_square = np.ones_like(square_retrieval_mask)[square_retrieval_mask]
    if data_dict['uncertainty'] is not None:
        unc_square = data_dict['uncertainty'][square_retrieval_mask]

    sens_square = np.ones_like(square_retrieval_mask)[square_retrieval_mask]
    if data_dict['sensitivity'] is not None:
        sens_square = data_dict['sensitivity'][square_retrieval_mask]
    
    # Collect intermediate 
    intermediate_results = []
    def plot_iteration(intermediate_result):
        #if intermediate_result.nit == 480:
        #    pdb.set_trace()
        intermediate_results.append(intermediate_result.copy())

    # Initial calculation of cost function
    init_cost = objective(init_params, data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                                    unc, sens, good_pixel_flags,
                                    nx, ny, dx, dy, x0_grid, y0_grid, pnt_list,
                                    [ip['plume_x_m'] for ip in init_plume_parameters_list],
                                    [ip['plume_y_m'] for ip in init_plume_parameters_list],
                                    plume_mean_locations_spacing_m,
                                    theta_deg_list, cfg['do_param_exp'])

    time_before_CRB, time_after_CRB = 0, 0
    ############################################################################################################################
    # Do main optimization
    time_before_min = time()
    opt = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b", maxiter=maxiter, tol = tol, options = local_opt_options)
    sol = opt.run(init_params, bounds = bounds,
                    mf_data = data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                    unc = unc,
                    sens = sens,
                    good_pixel_flags = good_pixel_flags, 
                    nx = nx,
                    ny = ny,
                    dx = dx,
                    dy = dy,
                    x0_grid = x0_grid,
                    y0_grid = y0_grid,
                    pnt_list = pnt_list,
                    plume_x_m_list = [ip['plume_x_m'] for ip in init_plume_parameters_list],
                    plume_y_m_list = [ip['plume_y_m'] for ip in init_plume_parameters_list],
                    plume_mean_locations_spacing_m = plume_mean_locations_spacing_m,
                    theta_deg_list = theta_deg_list,
                    do_exp = cfg['do_param_exp'])
    
    time_after_min = time()

    class Ret:
        def __init__(self, sol, log_param = False):
            params, state = sol
            local_x = np.array(params).copy()
            if log_param:
                local_x[0] = np.exp(local_x[0])
                local_x[1] = np.exp(local_x[1])
            n_params = len(local_x)
            n_params_per = n_params // n_plumes
            self.x = [local_x[plume_num*n_params_per:(plume_num+1)*n_params_per] for plume_num in np.arange(n_plumes)]
            self.nit = state.iter_num
            try:
                self.fun = state.fun_val
            except:
                self.fun = state.value
            try:
                self.message = state.status
                self.success = state.success
            except:
                self.message = ''
                self.success = -1
    r = Ret(sol, log_param = False) # We handle cfg['do_param_exp'] elsewhere!


    #    #jax_init_plume_parameters = {}
    #    #for k, val in init_plume_parameters.items():
    #    #    if k != 'plume_model':
    #    #        jax_init_plume_parameters[k] = val
    #    
    #    #retrieval_params_numeric = []
    #    #for i, k in enumerate(retrieval_params):
    #    #    retrieval_params_numeric.append(i)
    #    #    jax_init_plume_parameters[i] = jax_init_plume_parameters[k]


    #    #minimize_args = (data_dict['concentration_length_ort_ppmm'][retrieval_mask],
    #    #                 unc, sens, good_pixel_flags, 
    #    #                 nx, ny, dx, dy, x0_grid, y0_grid, pnt,
    #    #                 init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
    #    #                 theta_deg, cfg['do_param_exp'])
    #    #                 #jax_init_plume_parameters, retrieval_params_numeric)

    #    #init_params = [init_plume_parameters[key] for key in retrieval_params] + [0.]*len(d_1stmom_m) # Presence of d_1stmom_m does bendy
    #    n_bends = len(d_1stmom_m)
    #    init_params = [init_plume_parameters[key] for key in retrieval_params] + [0.]*n_bends # Presence of d_1stmom_m does bendy
    #    lb = lb + [-1*cfg['cross_plume_cross_length']*data_dict['gsd_ort_m']]*n_bends
    #    ub = ub + [   cfg['cross_plume_cross_length']*data_dict['gsd_ort_m']]*n_bends

    #    bounds = (jnp.array(lb), jnp.array(ub))
    #    maxiter = cfg['trust_constr_bendy_options']['maxiter']
    #    del cfg['trust_constr_bendy_options']['maxiter']
    #    tol = cfg['trust_constr_bendy_options']['tol']
    #    del cfg['trust_constr_bendy_options']['tol']
    #    opt = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b", maxiter=maxiter, tol = tol, options = cfg['trust_constr_bendy_options'])
    #    #opt = jaxopt.ScipyBoundedMinimize(fun=objective, method="l-bfgs-b", maxiter=maxiter)
    #    #opt = jaxopt.LBFGSB(fun=objective, maxiter=maxiter)
    #    #opt = jaxopt.ScipyBoundedMinimize(fun=objective, method='trust-constr', maxiter = maxiter, options = cfg['trust_constr_bendy_options'])
    #    #sol = opt.run(init_params, bounds=bounds, 
    #    sol = opt.run(init_params, bounds = bounds,
    #                  mf_data = data_dict['concentration_length_ort_ppmm'][retrieval_mask],
    #                  unc = unc,
    #                  sens = sens,
    #                  good_pixel_flags = good_pixel_flags, 
    #                  nx = nx,
    #                  ny = ny,
    #                  dx = dx,
    #                  dy = dy,
    #                  x0_grid = x0_grid,
    #                  y0_grid = y0_grid,
    #                  pnt = pnt,
    #                  plume_x_m = init_plume_parameters['plume_x_m'],
    #                  plume_y_m = init_plume_parameters['plume_y_m'],
    #                  theta_deg = theta_deg,
    #                  do_exp = cfg['do_param_exp'])
    #    #sol = opt.run(jnp.array(init_params), bounds = bounds,
    #    #              mf_data = data_dict['concentration_length_ort_ppmm'][retrieval_mask],
    #    #              unc = unc,
    #    #              sens = sens,
    #    #              good_pixel_flags = good_pixel_flags, 
    #    #              nx = jnp.array(nx),
    #    #              ny = jnp.array(ny),
    #    #              dx = jnp.array(dx),
    #    #              dy = jnp.array(dy),
    #    #              x0_grid = x0_grid,
    #    #              y0_grid = y0_grid,
    #    #              pnt = jnp.array(pnt),
    #    #              plume_x_m = jnp.array(init_plume_parameters['plume_x_m']),
    #    #              plume_y_m = jnp.array(init_plume_parameters['plume_y_m']),
    #    #              theta_deg = jnp.array(theta_deg),
    #    #              do_exp = cfg['do_param_exp'])
    #    class Ret:
    #        def __init__(self, sol, log_param = False):
    #            params, state = sol
    #            local_x = np.array(params).copy()
    #            if log_param:
    #                local_x[0] = np.exp(local_x[0])
    #                local_x[1] = np.exp(local_x[1])
    #            self.x = local_x
    #            self.nit = state.iter_num
    #            try:
    #                self.fun = state.fun_val
    #            except:
    #                self.fun = state.value
    #            try:
    #                self.message = state.status
    #                self.success = state.success
    #            except:
    #                self.message = ''
    #                self.success = -1
    #    

    #    #bounds = (list(jnp.array(lb)), list(jnp.array(ub)))
    #    #opt = jaxopt.ProjectedGradient(fun = objective, projection=jaxopt.projection.projection_box, 
    #    #                               maxiter = cfg['trust_constr_bendy_options']['maxiter'])
    #    #sol = opt.run(init_params,
    #    #              hyperparams_proj=bounds,
    #    #              # Parameters for objective function
    #    #              mf_data = data_dict['concentration_length_ort_ppmm'][retrieval_mask],
    #    #              unc = unc,
    #    #              sens = sens,
    #    #              good_pixel_flags = good_pixel_flags, 
    #    #              nx = nx,
    #    #              ny = ny,
    #    #              dx = dx,
    #    #              dy = dy,
    #    #              x0_grid = x0_grid,
    #    #              y0_grid = y0_grid,
    #    #              pnt = pnt,
    #    #              plume_x_m = init_plume_parameters['plume_x_m'],
    #    #              plume_y_m = init_plume_parameters['plume_y_m'],
    #    #              theta_deg = theta_deg,
    #    #              do_exp = cfg['do_param_exp'])
    #    #class Ret:
    #    #    def __init__(self, sol):
    #    #        params, state = sol
    #    #        self.x = np.array(params)
    #    #        self.nit = state.iter_num
    #    #        self.fun = 0
    #    #        self.message = 0
    #    #        self.success = 0

    #    #r = Ret(sol)

    print(r.nit)
    print(r.x)
    print('Calculating Cramer-Rao Bound...')

    time_before_CRB = time()
    if cfg['calc_CramerRaoBound']:
        obj_hess = jax.hessian(objective, argnums = 0)
        parameter_curvature = obj_hess(jnp.array(r.x),
                    mf_data = data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                    unc = unc,
                    sens = sens,
                    good_pixel_flags = good_pixel_flags, 
                    nx = nx,
                    ny = ny,
                    dx = dx,
                    dy = dy,
                    x0_grid = x0_grid,
                    y0_grid = y0_grid,
                    pnt_list = pnt_list,
                    plume_x_m_list = [ip['plume_x_m'] for ip in init_plume_parameters_list],
                    plume_y_m_list = [ip['plume_y_m'] for ip in init_plume_parameters_list],
                    plume_mean_locations_spacing_m = plume_mean_locations_spacing_m,
                    theta_deg_list = theta_deg_list,
                    do_exp = cfg['do_param_exp'])

                                        #data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                                        #unc, sens, good_pixel_flags,
                                        #nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                                        #init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
                                        #plume_mean_locations_spacing_m,
                                        #theta_deg, cfg['do_param_exp'])

        cq_vs_n = []
        all_vs_n = []
        try:
            for i in range(parameter_curvature.shape[0]-1):
                inv = np.linalg.inv(parameter_curvature[:i+1,:i+1])
                cq_vs_n.append(inv[0,0])
                all_vs_n.append(np.diag(inv)[:3])

            # The variance of C_Q (and a and exponent), increase each time we include more bend parameters
            # into inv(parameter_curvature[:i,:i]). Find the index of the first time it goes down. Use this
            # location as an indicator that the matrix inversion is going bad
            dif = np.diff(cq_vs_n) <= 0
            ind_max_valid = parameter_curvature.shape[0]
            if np.any(dif):
                ind_max_valid = (np.arange(parameter_curvature.shape[0]-2))[dif][0] + 1
            if ind_max_valid < 3:
                ind_max_valid = 3

        except:
            ind_max_valid = 3
        
        #for i in range(len(cq_vs_n)):
        #    print(i, all_vs_n[i])
        #print(f'ind_max_valid = {ind_max_valid}')

        debug = False
        if debug:
            tmpr     = Ret(sol, log_param=False)
            r_log = Ret(sol, log_param=True)
    
            print('')
            parameter_curvature_log = obj_hess(jnp.array(tmpr.x),
                                        data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                                        unc, sens, good_pixel_flags,
                                        nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                                        init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
                                        theta_deg_list, True)
            parameter_curvature_linear = obj_hess(jnp.array(r_log.x),
                                        data_dict['concentration_length_ort_ppmm'][retrieval_mask],
                                        unc, sens, good_pixel_flags,
                                        nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                                        init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
                                        theta_deg_list, False)

            try:
                var_lin = np.linalg.inv(np.array(parameter_curvature_linear)[:ind_max_valid, :ind_max_valid])
                print('linear params', r_log.x[:3])
                print('var_lin', np.diag(var_lin)[:3])
                cq = np.array([r_log.x[0] - np.sqrt(var_lin[0,0]), r_log.x[0], r_log.x[0] + np.sqrt(var_lin[0,0])]) * 3600
                print('lin C_Q range', cq)
                print('lin a range', r_log.x[1] - np.sqrt(var_lin[1,1]),
                                    r_log.x[1],
                                    r_log.x[1] + np.sqrt(var_lin[1,1]))
            except:
                print('linear inv failed')

            print('')
            try:
                var_log = np.linalg.inv(np.array(parameter_curvature_log)[:ind_max_valid, :ind_max_valid])
                print('log params', tmpr.x[:3])
                print('log var', np.diag(var_log)[:3])
                
                cq = np.array([np.exp(tmpr.x[0] - np.sqrt(var_log[0,0])), np.exp(tmpr.x[0]), np.exp(tmpr.x[0] + np.sqrt(var_log[0,0]))]) * 3600
                print('log C_Q range', cq)
                print('log a range', np.exp(tmpr.x[1] - np.sqrt(var_log[1,1])),
                                    np.exp(tmpr.x[1]),
                                    np.exp(tmpr.x[1] + np.sqrt(var_log[1,1])))
            except:
                print('log inv failed')
        
            #Jfwd = jax.jacfwd(objective, argnums = 0)
            #jf = Jfwd(jnp.array(r.x),
            #          data_dict['concentration_length_ort_ppmm'][retrieval_mask],
            #          unc, sens, good_pixel_flags,
            #          nx, ny, dx, dy, x0_grid, y0_grid, pnt,
            #          init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
            #          theta_deg, False)

            #Jrev = jax.jacrev(objective, argnums = 0)
            #jr = Jrev(jnp.array(r.x),
            #          data_dict['concentration_length_ort_ppmm'][retrieval_mask],
            #          unc, sens, good_pixel_flags,
            #          nx, ny, dx, dy, x0_grid, y0_grid, pnt,
            #          init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
            #          theta_deg, False)

            #Hess = jax.jacfwd(jax.jacrev(objective))
            #hs = Hess(jnp.array(r.x),
            #          data_dict['concentration_length_ort_ppmm'][retrieval_mask],
            #          unc, sens, good_pixel_flags,
            #          nx, ny, dx, dy, x0_grid, y0_grid, pnt,
            #          init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
            #          theta_deg, False)
        
        full_inv_status = 'success'
        try:
            pc_inv = np.linalg.inv(parameter_curvature[:ind_max_valid, :ind_max_valid])
        except:
            full_inv_status = 'failed'
            pc_inv = np.zeros_like(parameter_curvature)
        
        JJJ_inv_status = 'success'
        try:
            pc1on_inv = np.linalg.inv(parameter_curvature[1:ind_max_valid,1:ind_max_valid])
        except:
            JJJ_inv_status = 'failed'
            pc1on_inv = np.zeros_like(parameter_curvature[1:ind_max_valid,1:ind_max_valid])

        parameter_var = pc_inv
        Jww = parameter_curvature[0,0]
        JJJ = parameter_curvature[1:ind_max_valid,0] @ pc1on_inv @ parameter_curvature[0,1:ind_max_valid]


        time_after_CRB = time()
        
    ############################################################################################################################

    d_1stmom_m = np.array(r.x[len(retrieval_params):])

    print('')
    print('Final:')
    print(f'Success: {r.success}, nit: {r.nit}')
    print(f'{r.message}')
    print(f'Initial cost: {init_cost}, final cost: {r.fun}, init / final: {init_cost / r.fun}')

    # Diagnostic plots of convergence
    if cfg['plot_iterations'] and n_bends > 0:
        print('Plotting iterations...')
        if output_plot_basename is not None:
            cost = np.array([ir['fun'] for ir in intermediate_results])
            # Be sure to do the last one
            ir_step_list = [ir for i, ir in enumerate(intermediate_results) if i%cfg['plot_iterations_skip'] == 0] + \
                           [intermediate_results[-1]]
            for ir in ir_step_list:
                name = iter_pics + f'/plot_iter_{ir["nit"]:05d}.png'
                f = forward(ir['x'], 
                            nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                            init_plume_parameters['plume_x_m'], init_plume_parameters['plume_y_m'],
                            plume_mean_locations_spacing_m,
                            init_plume_parameters, retrieval_params, cfg['do_param_exp'])
                fig = make_plots(cfg, data_dict, ir, init_plume_parameters, f, 
                                retrieval_params, xs, ys, sens_square, unc_square, 
                                output_plot_name=name,
                                all_iter_cost = cost)
                plt.close(fig)

            f,a = plt.subplots(4,1, sharex = True, figsize = (7,13))
            a[0].plot(cost)
            ind_10pct = int(np.floor(len(cost)*0.1))
            a[0].set_ylim([cost[-1]*0.999, cost[ind_10pct]])
            a[1].plot([ir['x'][0]*3600 for ir in intermediate_results])
            a[2].plot([ir['x'][1] for ir in intermediate_results])
            a[3].plot([ir['x'][2] for ir in intermediate_results])
            [ax.set_title(rp) for ax, rp in zip(a, ['Cost'] + retrieval_params)]
            f.tight_layout()
            f.savefig(iter_pics + '/parameter_iterations.png', dpi = 150)
            plt.close(f)

    results = {}
    temp_theta_deg_list = []
    if n_plumes == 1:
        results['C_Q_kg_per_hr_mpers'] = np.exp(r.x[0][0]) * 3600 if cfg['do_param_exp'] else r.x[0][0] * 3600
    for plume_num in np.arange(n_plumes):
        x = r.x[plume_num]
        results[f'C_Q_kg_per_hr_mpers_{plume_num}'] = np.exp(x[0]) * 3600 if cfg['do_param_exp'] else x[0] * 3600
        results[f'C_Q_kg_per_s_mpers_{plume_num}'] = np.exp(x[0]) if cfg['do_param_exp'] else x[0]
        results[f'a_{plume_num}'] = np.exp(x[1]) if cfg['do_param_exp'] else x[1]
        results[f'exponent_{plume_num}'] = x[2]
        results[f'theta_deg_{plume_num}'] = x[3] if theta_deg_override is None else theta_deg_override
        results[f'full_retrieval_line_{plume_num}'] = x[len(retrieval_params):]
        temp_theta_deg_list.append(x[3])
    results[f'theta_deg_list'] = temp_theta_deg_list

    results['plot_data'] = data_dict['concentration_length_ort_ppmm'][xs, ys]
    plume_x_m_list = [ip['plume_x_m'] for ip in init_plume_parameters_list]
    plume_y_m_list = [ip['plume_y_m'] for ip in init_plume_parameters_list]
    f = forward_list(list(itertools.chain.from_iterable(r.x)), 
                     nx, ny, dx, dy, x0_grid, y0_grid, pnt_list,
                     plume_x_m_list, plume_y_m_list,
                     plume_mean_locations_spacing_m,
                     theta_deg_list, cfg['do_param_exp'])
    results['plot_model'] = np.array(f).copy().reshape((xs.stop - xs.start, ys.stop - ys.start))

    results['plume_id'] = plume_ids_str
    results['FID'] = data_dict['FID']
    results['success'] = r.success
    results['message'] = r.message
    results['n_iter'] = r.nit
    results['fun'] = r.fun
    results['initial_cost'] = init_cost
    #results[wind_corrected_Q_str.lower().replace(' ','_')+'_Q_kg_per_hr'] = results['Q_kg_per_s'] * 3600
    #results['C_Q_kg_per_hr_mpers'] = results['C_Q_kg_per_s_mpers'] * 3600
    results['gsd_ort_m'] = data_dict['gsd_ort_m']
    #results['wind_speed_m_per_s'] = wind_speed_m_per_s
    results['xs'] = xs
    results['ys'] = ys
    #results['cost_image'] = cost_image
    #results['cost_weighting'] = cost_weighting
    results['unc'] = unc_square.reshape((xs.stop - xs.start, ys.stop - ys.start))
    results['sens'] = sens_square.reshape((xs.stop - xs.start, ys.stop - ys.start))
    if retrieval_mask.dtype == bool:
        results['cost_per_pixel'] = r.fun / np.sum(retrieval_mask == True)
    
    near_lower = r.x < np.array(lb).reshape(n_plumes,-1) + 0.01*np.abs(np.array(lb).reshape(n_plumes,-1))
    near_upper = r.x > np.array(ub).reshape(n_plumes,-1) - 0.01*np.abs(np.array(ub).reshape(n_plumes,-1))
    results['ended_near_bound'] = np.logical_or(np.any(near_lower), np.any(near_upper))
    results['ended_near_lower_bound'] = near_lower
    results['ended_near_upper_bound'] = near_upper
    results['plume_time_UTC'] = data_dict['plume_time_UTC']

    results['rms_model_error'] = np.sqrt(np.nanmean((data_dict['concentration_length_ort_ppmm'][retrieval_mask] - f)**2))

    for key in data_dict.keys():
        results[key] = data_dict[key]
    del results['concentration_length_ort_ppmm']

    if cfg['calc_CramerRaoBound']:
        results['n_params'] = parameter_curvature.shape[0]
        results['ind_max_valid'] = ind_max_valid
        results['Jww'] = Jww
        results['JJJ'] = JJJ
        results['full_inv_status'] = full_inv_status
        results['JJJ_inv_status'] = JJJ_inv_status
        results['Jww_std_inv'] = 1/np.sqrt(Jww)
        results['Jww_JJJ_std_inv'] = 1/np.sqrt(Jww - JJJ)

        cq = np.array([r.x[0] - np.sqrt(parameter_var[0,0]), 
                       r.x[0],
                       r.x[0] + np.sqrt(parameter_var[0,0])])
        if cfg['do_param_exp']:
            cq = np.exp(cq)
        cq = cq * 3600

        if Jww == JJJ:
            results['C_Q_kg_per_hr_mpers_std'] = 'Jww = JJJ'
        elif JJJ == 0.:
            results['C_Q_kg_per_hr_mpers_std'] = 'JJJ = 0'
        elif JJJ > Jww:
            results['C_Q_kg_per_hr_mpers_std'] = 'JJJ > Jww'
        else:
            results['C_Q_kg_per_hr_mpers_std'] = cq[2]-cq[1]
        
        results['C_Q_kg_per_hr_mpers_low'] = cq[0]
        results['C_Q_kg_per_hr_mpers_mid'] = cq[1]
        results['C_Q_kg_per_hr_mpers_high'] = cq[2]

        aa = np.array([r.x[1] - np.sqrt(parameter_var[1,1]), 
                       r.x[1],
                       r.x[1] + np.sqrt(parameter_var[1,1])])
        if cfg['do_param_exp']:
            aa = np.exp(aa)
        results['a_low'] = aa[0]
        results['a_mid'] = aa[1]
        results['a_high'] = aa[2]

        results['exponent_std'] = np.sqrt(parameter_var[2,2])

        results['parameter_curvature'] = parameter_curvature
        results['parameter_var'] = parameter_var

    if show_plots:
        #fig, fig_cost = make_plots(data_dict, r, 
        fig = make_plots(cfg, data_dict, r, 
                      init_plume_parameters,
                      f,
                      retrieval_params,
                      xs, ys,
                      sens_square, unc_square,
                      pnt_list, output_plot_name=output_plot_basename)
        results['fig'] = fig
        #results['fig_cost'] = fig_cost

    time_after_plots = time()
    print(f'Total time: {time() - start_time:.2f}, minimize time: {time_after_min - time_before_min:.2f}, plot time: {time_after_plots - time_after_min:.2f}, start to min: {time_before_min - start_time:.2f}, CRB_time: {time_after_CRB - time_before_CRB:.2f}')

    if close_plot and show_plots:
        plt.close(fig)
        #plt.close(fig_cost)

    return results

def draw_x_on_pixel(ax, row, col, color='red', thickness=0.25):
    """
    Draw a filled 'X' inside the single pixel at (row, col) for an imshow image.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the image produced by imshow.
    row, col : int
        Pixel indices (row=y index, col=x index).
    color : str or tuple
        Face color of the X.
    thickness : float
        Thickness of the diagonal bars as a fraction of the pixel size (0..1).
        For example, 0.25 means each diagonal rectangle is ~25% of the pixel's width.
    """
    # Safety clamp
    t = max(0.01, min(0.49, thickness))  # keep within reasonable bounds
    
    # Pixel bounding box in data coordinates for default imshow (extent=None, origin='upper'):
    # Each pixel spans [col-0.5, col+0.5] in x and [row-0.5, row+0.5] in y.
    x0, x1 = col - 0.5, col + 0.5
    y0, y1 = row - 0.5, row + 0.5
    bbox = transforms.Bbox.from_extents(x0, y0, x1, y1)
    to_pixel = transforms.BboxTransformTo(bbox)

    # Define two diagonal rectangles in a unit square [0,1]x[0,1].
    # Diagonal 1: from (0,0) to (1,1), perpendicular vector (1,-1)/sqrt(2)
    # Diagonal 2: from (0,1) to (1,0), perpendicular vector (1, 1)/sqrt(2)
    import math
    p1 = np.array([1.0, -1.0]) / math.sqrt(2)
    p2 = np.array([1.0,  1.0]) / math.sqrt(2)
    half_w = t / 2.0

    # Rectangle around diagonal 1
    a1 = np.array([0.0, 0.0])
    b1 = np.array([1.0, 1.0])
    r1_pts = np.array([
        a1 - half_w * p1,
        b1 - half_w * p1,
        b1 + half_w * p1,
        a1 + half_w * p1
    ])

    # Rectangle around diagonal 2
    a2 = np.array([0.0, 1.0])
    b2 = np.array([1.0, 0.0])
    r2_pts = np.array([
        a2 - half_w * p2,
        b2 - half_w * p2,
        b2 + half_w * p2,
        a2 + half_w * p2
    ])

    # Build a single Path with two subpaths (polygons) and use even-odd fill to combine
    verts = []
    codes = []

    def add_rect(pts):
        verts.extend(pts.tolist())
        codes.extend([path.Path.MOVETO] + [path.Path.LINETO]*(len(pts)-1))
        # Close the polygon
        verts.append(pts[0].tolist())
        codes.append(path.Path.CLOSEPOLY)

    add_rect(r1_pts)
    add_rect(r2_pts)

    x_path = path.Path(verts, codes)

    # Create patch that maps unit-square path to the pixel's bbox
    patch = patches.PathPatch(
        x_path,
        facecolor=color,
        edgecolor='none',     # no stroke; fill scales perfectly with the pixel size
        transform=to_pixel + ax.transData,
        fill=True
    )
    ax.add_patch(patch)
    return patch

def make_plots(cfg, data_dict, r, init_plume_parameters, f, retrieval_params,
               xs, ys, sens_square, unc_square, pnt_list, cost_image = None, cost_weighting = None,
               output_plot_name = None, all_iter_cost = None):
    try:
        nit, fun, rx = r.nit, r.fun, r.x
    except:
        nit, fun, rx = r['nit'], r['fun'], r['x']
    plot_data = data_dict['concentration_length_ort_ppmm'][xs, ys].T
    plot_data[~np.isfinite(plot_data)] = 0
    #f, _ = forward(r['x'], init_plume_parameters, pm_plot, retrieval_params, do_full_scene = False, d_1stmom_m = None, do_derivs = True)
    f = f.reshape((xs.stop - xs.start, ys.stop - ys.start)).T
    fig = plt.figure(figsize = (15,8))
    axes = ImageGrid(fig, 111, nrows_ncols=(2,2), axes_pad=0.10, share_all=True, cbar_location="right", cbar_mode="single", cbar_size="4%", cbar_pad=0.50)
    for ax in axes:
        ax.set_navigate(True)           # now toolbar works
    for cax in axes.cbar_axes:
        cax.set_navigate(False)
    vmin, vmax = 0, 1500
    ext = [xs.start-0.5, xs.stop-0.5, ys.start-0.5, ys.stop-0.5]
    im0 = axes[0].imshow(plot_data, origin = 'lower',
                         vmin = vmin, vmax = vmax, cmap = 'inferno', extent=ext)
    im1 = axes[3].imshow(f * sens_square.reshape((xs.stop - xs.start, ys.stop - ys.start)).T, origin = 'lower',
                         vmin = vmin, vmax = vmax, cmap = 'inferno', extent=ext)
    im1 = axes[2].imshow(f, origin = 'lower',
                         vmin = vmin, vmax = vmax, cmap = 'inferno', extent=ext)
    im2 = axes[1].imshow(plot_data, origin = 'lower',
                         vmin = vmin, vmax = vmax, cmap = 'gray', extent=ext)
    alphas = (f>200.0) * 0.8
    axes[1].imshow(f * sens_square.reshape((xs.stop - xs.start, ys.stop - ys.start)).T, origin = 'lower',
                   vmin = vmin, vmax = vmax, cmap = 'inferno', extent=ext, alpha = alphas)

    #im1 = axes[1].imshow(forward(r.x, init_plume_parameters, pm, retrieval_params, do_full_scene = False).T, origin = 'lower', vmin = vmin, vmax = vmax, cmap = 'inferno', extent=[xs.start,xs.stop,ys.start,ys.stop])
    #im2 = axes[2].imshow(plot_data - forward(r.x, init_plume_parameters, pm, retrieval_params, do_full_scene = False).T, origin = 'lower', vmin = vmin, vmax = vmax, cmap = 'inferno', extent=[xs.start,xs.stop,ys.start,ys.stop])

    for plume_ort_x, plume_ort_y, pso_x, pso_y in zip(data_dict['plume_ort_x_list'], data_dict['plume_ort_y_list'], data_dict['pseudo_origin_ort_x_list'], data_dict['pseudo_origin_ort_y_list']):
        axes[0].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
        axes[1].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
        axes[2].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
        axes[3].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
        axes[0].plot(pso_x, pso_y, 'o', color = 'white')
        axes[1].plot(pso_x, pso_y, 'o', color = 'white')
        axes[2].plot(pso_x, pso_y, 'o', color = 'white')
        axes[3].plot(pso_x, pso_y, 'o', color = 'white')
    axes[0].cax.colorbar(im0)

    for ax in axes:
        for pnt in pnt_list:
            i, j = np.unravel_index(pnt, f.T.shape)
            draw_x_on_pixel(ax, j[0]+ys.start, i[0]+xs.start)
    
    if all_iter_cost is not None:
        def convertx_to_axes_units(x_range, v, axis):
            return (v - x_range[0]) / (x_range[1] - x_range[0]) * axis.viewLim.width + axis.viewLim.x0
        axes2 = [a.twinx() for a in axes]
        for a in axes2:
            a.plot(convertx_to_axes_units([0,len(all_iter_cost)], np.arange(len(all_iter_cost)),a), all_iter_cost, color = 'green')
            #a.plot(convertx_to_axes_units([0,len(all_iter_cost)], r['nit']-1,a), r['fun'], 'o', color = 'green')
            a.plot(convertx_to_axes_units([0,len(all_iter_cost)], nit-1,a), fun, 'o', color = 'green')
        axes2[0].get_yaxis().set_visible(False), axes2[2].get_yaxis().set_visible(False)
        axes2[1].tick_params(axis = 'y', colors = 'green'), axes2[3].tick_params(axis = 'y', colors = 'green')
        ind_10pct = int(np.floor(len(all_iter_cost)*0.1))
        _ = [a.set_ylim([all_iter_cost[-1], all_iter_cost[ind_10pct]]) for a in axes2]

    svals = ''
    tmp = retrieval_params.copy()
    tmp[tmp.index('C_Q_kg_per_s_mpers')] = 'C_Q'
    s_truth = ''
    if 'controlled_release' in cfg.keys():
        if cfg['controlled_release']:
            plume_id = data_dict['plume_ids'][0]
            if plume_id.endswith('_CH4'):
                plume_id = plume_id.replace('_CH4','')
            df = pd.read_csv(cfg['controlled_release_truth']).set_index(['plume_id'])
            Q_truth = df.loc[plume_id]['Q_truth']
            w_truth = df.loc[plume_id]['MeanWndSpd_10meter(m/s)_smoothed']
            time_truth = df.loc[plume_id]['plume_time_UTC']
            Q_est = np.exp(r.x[0][retrieval_params.index('C_Q_kg_per_s_mpers')]) * w_truth * 3600
        
            s_truth = f'{plume_id}\n' + \
                    f'{time_truth}\n' + \
                    f'True Q: {Q_truth:.0f} kg/hr\n' + \
                    f'Est. Q: {Q_est:.0f} kg/hr\n' + \
                    f'Wind speed: {w_truth:.0f} m/s\n\n'
        

    s0 = f'N_iter: {nit}\n'
    s = ''
    for x in rx:
        for retrieval_param, xval in zip(tmp, x):
            stemp = f'{retrieval_param}: {xval:0.4f}'
            svals = svals + stemp + '\n'
        s = s + svals
    s0 = s_truth + s0 + s
    fig.text(0.02, 0.98, s0, fontsize=14, transform=plt.gcf().transFigure, va = 'top', ha = 'left')

    #results['fig'] = fig
    return_val = fig

    if cost_image is not None:
        fig_cost = plt.figure(figsize = (13,8))
        #axes_cost = ImageGrid(fig_cost, 111, nrows_ncols=(1,2), axes_pad=0.10, share_all=True, cbar_location="right", cbar_mode="single", cbar_size="7%", cbar_pad=0.15)
        axes_cost = ImageGrid(fig_cost, 111, nrows_ncols=(1,2), axes_pad = (0.45, 0.15), cbar_location = 'right', cbar_mode='each', cbar_size="7%", cbar_pad='2%')
        vmin, vmax = np.percentile(cost_image, [1,99])
        im0 = axes_cost[0].imshow(cost_image.T, origin = 'lower', vmin = vmin, vmax = vmax, cmap = 'inferno', extent=[xs.start,xs.stop,ys.start,ys.stop])
        im1 = axes_cost[1].imshow((np.ones_like(cost_image)*cost_weighting).T, origin = 'lower', cmap = 'inferno', extent=[xs.start,xs.stop,ys.start,ys.stop])
        for plume_ort_x, plume_ort_y in zip(data_dict['plume_ort_x_list'], data_dict['plume_ort_y']):
            axes_cost[0].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
            axes_cost[1].plot(plume_ort_x, plume_ort_y, '-', color = 'white')
        axes_cost[0].cax.colorbar(im0)
        axes_cost[1].cax.colorbar(im1)
        #axes_cost[0].cax.toggle_label(True)
        #axes_cost[1].cax.toggle_label(True)
        #results['fig_cost'] = fig_cost
        return_val = (fig, fig_cost)

    #results['unc'] = unc_square.reshape((xs.stop - xs.start, ys.stop - ys.start))
    #results['sens'] = sens_square.reshape((xs.stop - xs.start, ys.stop - ys.start))

    if output_plot_name is not None:
        fig.savefig(output_plot_name, dpi = 250)
        #pickle.dump(fig, open(output_plot_basename + '.pickle', 'wb'))
    return return_val


def check_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' does not exist")
    return path

def main(input_yaml_file, plume_ids):
    cfg = yaml.safe_load(open(input_yaml_file, 'r'))
    if not os.path.exists(cfg['output_path']):
        os.mkdir(cfg['output_path'])
    shutil.copy2(input_yaml_file, cfg['output_path'] + '/' + '.'.join(input_yaml_file.split('/')[-1].split('.')[:-1]) + '_COPY.yaml')

    if cfg['show_plots_to_screen']:
        plt.ion()

    run_one(cfg, plume_ids)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('input_yaml_file', type=check_path, help='Path to the input yaml file')
    parser.add_argument(
        '-p', '--plume_ids',
        nargs='+',
        type=str,
        required=True,
        help='Space-separated list of plume IDs (always returned as a list)'
    )
    args = parser.parse_args()

    main(args.input_yaml_file, args.plume_ids)

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']

   #run_all(retrieval_params, 'AV3', run_list_csv_filename='/store/jfahlen/av3_wyoming_Sept2024/plume_ids.csv', output_path = '/store/jfahlen/av3_wyoming_Sept2024/2d_inv_full_500iter_no_cost/', 
   #         use_saved_data_dict = False, saved_data_dict_location = None, parallel = False, plume_json_filename = '/store/jfahlen/av3_wyoming_Sept2024/plume_list.json')

    #run_all(retrieval_params, 'AV3', run_list_csv_filename='/store/jfahlen/av3_casa_grande_Nov2024/plume_ids.csv', output_path = '/store/jfahlen/av3_casa_grande_Nov2024/2d_inv_full_500iter_no_cost/', 
    #        use_saved_data_dict = False, saved_data_dict_location = None, parallel = False, plume_json_filename = '/store/jfahlen/av3_casa_grande_Nov2024/plume_list.json')

    #run_all(retrieval_params, 'AV3', run_list_csv_filename='/beegfs/scratch/jfahlen/av3_flight_info/av3_controlled_release_pseudo_origin.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_av3_bendy_full/', 
    #        use_saved_data_dict = True, saved_data_dict_location = '/beegfs/scratch/jfahlen/saved_plume_estimate_data_dicts_JFMF/', parallel = True)

    #run_all(retrieval_params, 'EMIT', '/beegfs/scratch/jfahlen/emit_case_lists/20240221_plume_list_CM_with_windspeeds.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_emit300_bendy_full/', 
    #        use_saved_data_dict = False, saved_data_dict_location = None, parallel = True)

    # Weird EMIT list from Willow
    #run_all(retrieval_params, 'EMIT', '/scratch/colemanr/plume_emission_rate/EMIT_300_comparisons/emit_300_reanalysis.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_emit300_bendy/', 
    #        use_saved_data_dict = False, saved_data_dict_location = None, parallel = True)

    #run_all(retrieval_params, 'ANG', '/beegfs/scratch/jfahlen/ang_data/controlled_release_Sep2018/flight_data.csv', '/beegfs/scratch/jfahlen/plume_estimates_ang2018_bendy/', 
    #        use_saved_data_dict = True, saved_data_dict_location = '/beegfs/scratch/jfahlen/ang_data/controlled_release_Sep2018/', parallel = True)



    # AV3 4/22/2024
    #retrieval_params = ['Q_kg_per_s', 'exponent', 'a', 'theta_deg']
    #run_all(retrieval_params, run_list_csv_filename='/scratch/colemanr/av3/controlled_release_info_out.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_av_limits/', parallel = True)

    #retrieval_params = ['Q_kg_per_s', 'exponent', 'a', 'theta_deg']
    #run_all(retrieval_params, run_list_csv_filename='/scratch/colemanr/av3/controlled_release_info_out.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_av_limits_other_bounds/', parallel = True)

    #retrieval_params = ['Q_kg_per_s', 'exponent', 'a', 'theta_deg']
    #run_all(retrieval_params, run_list_csv_filename='/scratch/colemanr/av3/controlled_release_info_out.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_av_limits_full_range/', parallel = True)

    #retrieval_params = ['Q_kg_per_s', 'exponent', 'a', 'theta_deg']
    #run_all(retrieval_params, run_list_csv_filename='/scratch/colemanr/av3/controlled_release_info_out.csv', output_path = '/beegfs/scratch/jfahlen/plume_estimates_av_limits_fixed/', parallel = True)


    # EMIT
    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_limits_full_range/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_limits_full_range_200iter/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_limits_fixed/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_limits_other_bounds/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_limits/')

    #run_all(retrieval_params, 'EMIT', '/home/jfahlen/20240221_plume_list_CM.csv', '/beegfs/scratch/jfahlen/plume_estimates_emit/', parallel = True)
    #run_all(retrieval_params, 'EMIT', '/scratch/colemanr/plume_emission_rate/EMIT_300_comparisons/emit_300_reanalysis.csv', '/beegfs/scratch/jfahlen/plume_estimates_emit_reanlysis/', parallel = True)
    #run_all(retrieval_params, 'EMIT', '/scratch/colemanr/plume_emission_rate/EMIT_300_comparisons/emit_300_reanalysis.csv', '/beegfs/scratch/jfahlen/plume_estimates_emit_reanlysis/'', parallel = True)






    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_search_origin/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'a']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_fixed_exponent/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s', 'exponent']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_fixed_a/')

    #retrieval_params = ['theta_deg', 'Q_kg_per_s']
    #run_all(retrieval_params, parallel = True, output_path = '/beegfs/scratch/jfahlen/plume_estimates_fixed_exponent_and_a/')

    #run_all(parallel = False)
