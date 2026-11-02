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
import scipy.optimize
import scipy.stats
import pickle

from PIL import Image
import matplotlib.pyplot as plt
import pl

def compute_cross_plume_slices(data_dict, n_slices_in, plume_num,
                               n_slices_per_figure = 10, cross_length = 21, full_retrieval_line = None, 
                               output_plot_basename = None, show_plots = True, plot_name_tag = ''):
    if type(data_dict) == str:
        data_dict = pickle.load(open(data_dict, 'rb'))
    d = data_dict['plot_data']
    m = data_dict['plot_model']# * data_dict['sens']
    m[~np.isfinite(m)] = 0
    ms = m * data_dict['sens']

    #################################################################################################################################################
    # Get retrieved plume origin pixel index in d and m
    xoff = data_dict['xoff'] if 'xoff' in data_dict.keys() else 0
    yoff = data_dict['yoff'] if 'yoff' in data_dict.keys() else 0

    pso = data_dict['pseudo_origin_ort_x_list'][plume_num] + xoff - data_dict['xs'].start, \
          data_dict['pseudo_origin_ort_y_list'][plume_num] + yoff - data_dict['ys'].start
    #################################################################################################################################################
    plume_id = data_dict['plume_ids'][plume_num]
    theta_deg = data_dict[f'theta_deg_{plume_num}']
    a_param = data_dict[f'a_{plume_num}']
    exponent = data_dict[f'exponent_{plume_num}']
    C_Q_kg_per_s_mpers = data_dict[f'C_Q_kg_per_s_mpers_{plume_num}']
    gsd = data_dict['gsd_ort_m']

    # Coordinates of cross wind slices
    lx, ly = np.arange(cross_length) - cross_length//2, np.zeros(cross_length)
    lxr, lyr = rotate_vector(theta_deg + 90, np.vstack((lx,ly)))
    lx_m, ly_m = lx * gsd, ly * gsd

    # Coordinates of along wind axis
    x_alongwind_list = np.arange(n_slices_in)
    along_points = rotate_vector(theta_deg, np.vstack((x_alongwind_list, np.zeros(n_slices_in)))).T
    
    # Keep only those that are inside the data box
    xin = (along_points[:,0]+pso[0] > 0)   &   (along_points[:,0]+pso[0] < d.shape[0])
    yin = (along_points[:,1]+pso[1] > 0)   &   (along_points[:,1]+pso[1] < d.shape[1])
    inside = xin & yin
    n_slices = np.sum(inside)
    x_alongwind_list = x_alongwind_list[inside]
    along_points = along_points[inside, :]

    # Collect data
    d_0thmom_list, m_0thmom_list = [], []
    d_1stmom_list, m_1stmom_list = [], []
    d_2ndmom_list, m_2ndmom_list = [], []
    plot_data_list = []
    sigma_y_list = []
    ierr_list = []

    # Loop over each cross wind slice
    for n, (x_alongwind, along_point) in enumerate(zip(x_alongwind_list, along_points)):
        
        # Coordinates of cross wind slice at the along wind position along_point
        y_crosswind = np.vstack((lxr + pso[0],lyr + pso[1])).T + along_point

        # Interpolated cross wind slices through the data and model
        d_slice  = scipy.ndimage.map_coordinates(d,  y_crosswind.T, order = 3)
        m_slice  = scipy.ndimage.map_coordinates(m,  y_crosswind.T, order = 3)
        ms_slice = scipy.ndimage.map_coordinates(ms, y_crosswind.T, order = 3)
        s_slice  = scipy.ndimage.map_coordinates(data_dict['sens'], y_crosswind.T, order = 3)

        # Find the moments of the data and model along the cross wind slice
        tmpd, tmpm = np.where(d_slice < 0, 0, d_slice), np.where(m_slice < 0, 0, m_slice)
        stmpd, stmpm = np.sum(tmpd), np.sum(tmpm)
        d1 = 0 if stmpd == 0 else np.sum(lx_m * tmpd) / stmpd
        m1 = 0 if stmpm == 0 else np.sum(lx_m * tmpm) / stmpm
        d2 = 0 if stmpd == 0 else np.sqrt(np.sum((lx_m-d1)**2 * tmpd) / stmpd)
        m2 = 0 if stmpm == 0 else np.sqrt(np.sum((lx_m-m1)**2 * tmpm) / stmpm)
        m_1stmom_list.append(m1),  m_2ndmom_list.append(m2)

        #################################################################################################################################################
        # Find the mean and width of a Gaussian fit at each cross wind slice
        def gaussian(x, amplitude, mean, stddev):
            r = amplitude / np.sqrt(2*np.pi) / np.abs(stddev) * np.exp(-0.5 * (x - mean)**2 / stddev**2)
            return r.astype('float64')
        # Alternative that includes the sensitivity, but this is not well tested and doesn't seem to work well.
        #def gaussian_sens(x, amplitude, mean, stddev):
        #    r = s_slice * amplitude / np.sqrt(2*np.pi) / np.abs(stddev) * np.exp(-0.5 * (x - mean)**2 / stddev**2)
        #    r = amplitude / np.sqrt(2*np.pi) / np.abs(stddev) * np.exp(-0.5 * (x - mean)**2 / stddev**2)
        #    return r.astype('float64')

        # Ensure that fit stays within the cross_length on either side
        # The width should be larger than at least 1 GSD
        bounds = ([0,  -2*cross_length*gsd, gsd], 
                  [1e8, 2*cross_length*gsd, 1e8])

        # Initial values
        init_d1 = lx_m[np.argmax(d_slice)] # Start at peak of data, but stay within the bounds
        if d1 > 2*cross_length*gsd:
            init_d1 = 2*cross_length*gsd - 0.1
        elif d1 < -2*cross_length*gsd:
            init_d1 = -2*cross_length*gsd + 0.1
        p0 = [np.max(m_slice), init_d1, 2*gsd]
        
        try:
            popt, _, infodict, mesg, ierr = scipy.optimize.curve_fit(gaussian, lx_m.astype('float64'), d_slice.astype('float64'), 
                                            p0 = p0, bounds = bounds, full_output = True)
        except:
            ierr = -1
            popt = [np.max(d_slice), d1, np.abs(d2)]

        d_0thmom_list.append(popt[0] / np.sqrt(2*np.pi) / np.abs(popt[2]))
        d_1stmom_list.append(popt[1])
        d_2ndmom_list.append(np.abs(popt[2]))
        ierr_list.append(ierr)
        #################################################################################################################################################

        # Evaluate the 2d inversion's plume model to get the modeled width and amplitude
        sigma_y = a_param *  ( (0+np.abs(x_alongwind * gsd)) / 1000.)**exponent
        sigma_y_list.append(sigma_y)
        k = (1.0/1e6)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0) # scaling factor from ppm m to kg / m^2 CH4
        tmp = 0
        if sigma_y > 0:
            tmp = C_Q_kg_per_s_mpers / k / np.sqrt(2*np.pi) / sigma_y
        m_0thmom_list.append(tmp)
        
        # Collect data so we can plot it below
        lx_fine = np.linspace(lx_m[0], lx_m[-1], 1000) # Evaluate the fit cross wind at fine resolution for plotting
        plot_data_list.append((lx_m, d_slice, m_slice, ms_slice, lx_fine, gaussian(lx_m, *popt), gaussian(lx_fine, *popt)))
    

    #################################################################################################################################################
    # Interpolate through slices whose Gaussian fit failed
    ierr = np.array(ierr_list)
    valid = np.logical_and(ierr > 0, ierr < 5) # These values are valid according to documentation
    valid = np.logical_and(valid, ierr != 1)
    
    # Ensure that the first and last have valid values, even if they have to be replaced with 0
    if valid[0] == False:
        d_1stmom_list[0] = 0
        valid[0] = True
    if valid[-1] == False:
        d_1stmom_list[-1] = 0
        valid[-1] = True

    xv = np.arange(len(d_1stmom_list))[valid]
    yv = np.array(d_1stmom_list)[valid]
    d_1stmom_m = np.interp(np.arange(len(d_1stmom_list)), xv, yv)
    #################################################################################################################################################

    # Iteratively remove outliers and smooth the first moments
    d_1stmom_smoothed_m = filter_loop(d_1stmom_m, np.array(d_2ndmom_list))

    # Smooth the width
    d_2ndmom_m = np.array(d_2ndmom_list)
    sgf_len = np.min([len(d_2ndmom_m), 9])
    d_2ndmom_smoothed_m = np.abs(scipy.signal.savgol_filter(scipy.ndimage.maximum_filter(d_2ndmom_m,5), sgf_len, 2))

    x, y = rotate_vector(theta_deg, np.vstack((x_alongwind_list, np.array(d_1stmom_m)/gsd)))
    xsmo, ysmo = rotate_vector(theta_deg, np.vstack((x_alongwind_list, d_1stmom_smoothed_m/gsd)))
    if full_retrieval_line is not None:
        try:
            frl_smo = scipy.signal.savgol_filter(full_retrieval_line, 19, 2)
        except:
            frl_smo = full_retrieval_line.copy()
        #frl_smo = scipy.signal.savgol_filter(full_retrieval_line, 3, 2)
        #frl_smo = scipy.signal.savgol_filter(frl_smo, 5, 2)
        #frl_smo = scipy.signal.savgol_filter(frl_smo, 3, 2)
        frl_smo_pix = frl_smo/gsd
        if frl_smo_pix.shape[0] > len(x_alongwind_list):
            frl_smo_pix = frl_smo_pix[:len(x_alongwind_list)]
        xf, yf = rotate_vector(theta_deg, np.vstack((x_alongwind_list, frl_smo_pix)))
    

    if show_plots:
        xvec = np.array(x_alongwind_list) * gsd
        # Now plot it all
        # 2d plot showing the plume and plume model with the moments plotted
        f2d, a2d = plt.subplots(1,2, figsize = (12,6), sharex = True, sharey = True)
        nx, ny = d.shape
        im = a2d[0].imshow(d.T, origin = 'lower', cmap = 'plasma', vmin = 0, vmax = 1500)
        im = a2d[1].imshow(m.T, origin = 'lower', cmap = 'plasma', vmin = 0, vmax = 1500)
        a2d[0].plot(along_points[:,0] + pso[0], along_points[:,1] + pso[1], '-', color = 'gray')
        for ax, ay in zip(along_points[::n_slices_per_figure,0] + pso[0], along_points[::n_slices_per_figure,1] + pso[1]):
            a2d[0].plot(ax + lxr, ay + lyr, color = 'gray')

        a2d[0].plot(along_points[::n_slices_per_figure,0] + pso[0], along_points[::n_slices_per_figure,1] + pso[1], 'o', color = 'gray', markersize = 10)
        a2d[1].plot(along_points[:,0] + pso[0], along_points[:,1] + pso[1], '-', color = 'gray')
        a2d[1].plot(along_points[::n_slices_per_figure,0] + pso[0], along_points[::n_slices_per_figure,1] + pso[1], 'o', color = 'gray', markersize = 10)

        #pdb.set_trace()
        for xpos, ypos, ss in zip(along_points[::n_slices_per_figure,0] + pso[0], along_points[::n_slices_per_figure,1] + pso[1], [f'{xx}: {xx*n_slices_per_figure*gsd:2.1f}m' for xx in np.arange(n_slices//n_slices_per_figure)]):
            ang = (theta_deg + 90) % 360
            if ang > 180:
                ang = (theta_deg + 90) % 360 - 180
            a2d[0].text(xpos, ypos, ss, ha='left', va='bottom', fontsize = 20, color = 'gray', rotation = ang, rotation_mode = 'anchor')
            a2d[1].text(xpos, ypos, ss, ha='left', va='bottom', fontsize = 20, color = 'gray', rotation = ang, rotation_mode = 'anchor')

        # The cross wind slices are plotted in figures n_slices_per_figure at a time so the matplotlib figure doesn't get too big
        f_list = [plt.subplots(n_slices_per_figure//2, 2, sharex = True, sharey = True, figsize = (9, 13)) for i in range((n_slices-1)//n_slices_per_figure + 1)]
        f_list = [(f, np.array(a).T.flatten()) for f,a in f_list]

        pmin, pmax = np.percentile(np.array([p[1] for p in plot_data_list]), [2,100])
        p_len = (pmax - pmin)
        for n, (x_alongwind, plot_data) in enumerate(zip(x_alongwind_list, plot_data_list)):
            f, axes = f_list[n//n_slices_per_figure]
            f.suptitle(f'Plot Number {n//n_slices_per_figure}')
            a = axes[n%n_slices_per_figure]

            lx_m, d_slice, m_slice, ms_slice, lx_fine, g, g_fine = plot_data
            #a.plot(lx_m, d_slice, label = 'd', color = 'C0')
            #a.plot(lx_m, m_slice, label = 'm', color = 'C1')
            #a.plot(lx_m, ms_slice, '--', label = 'ms', color = 'C1')
            a.plot(lx_m, d_slice, label = 'd', color = 'black', linewidth=2)
            #a.plot(lx_m, m_slice, label = 'm', color = 'C1')
            a.plot(lx_m, ms_slice, label = 'ms', color = 'C0', linewidth=3)
            dist_str = f'{x_alongwind*gsd:2.1f} m'
            a.text(0.95, 0.95, dist_str, ha='right', va='top', fontsize = 12, transform=a.transAxes)
            
            #a.plot(lx_m, g, 'o', color = 'C2')
            #a.plot(lx_fine, g_fine, '--', label = 'mm', color = 'C2')
            #a.plot(lx_m, g, '--', label = 'mm', color = 'C2')
            #a.plot(lx_fine, g_fine, label = 'mm', color = 'C2')
            a.set_ylim([pmin, pmax + p_len*0.1])
            #a.set_xlim([lx_fine[0] * 0.4, lx_fine[-1] * 0.4])
            a.set_xlim([-150,150])
            a.tick_params(axis='both', which='major', labelsize=16)

        def set_range(d_list, perc = [5,95], fac = 0.3):
            lo, hi = np.percentile(np.array(d_list), perc)
            r = hi - lo
            if not np.isfinite(r):
                return np.nanmin(np.array(d_list)), np.nanmax(np.array(d_list))
            return lo - r*fac, hi + r*fac

        fmom, amom = plt.subplots(3,1, figsize = (13,13), sharex = True)
        amom[0].set_title('Amplitude')
        amom[0].plot(xvec, d_0thmom_list, label = 'Data')
        amom[0].plot(xvec, m_0thmom_list, label = 'Model')
        amom[0].set_ylim(set_range(d_0thmom_list))

        amom[1].set_title('1st Moment')
        amom[1].plot(xvec, d_1stmom_list, label = 'Data')
        amom[1].plot(xvec, d_1stmom_smoothed_m, label = 'Smoothed Data')
        if full_retrieval_line is not None:
            amom[1].plot(xvec, frl_smo_pix, label = '2D Inversion mean')
        amom[1].set_ylim(set_range(d_1stmom_list))

        amom[2].set_title('2nd Moment')
        amom[2].plot(xvec, d_2ndmom_list, label = 'Data')
        amom[2].plot(xvec, d_2ndmom_smoothed_m, label = 'Data Smoothed', color = 'C0', linestyle = '--')
        amom[2].plot(xvec, m_2ndmom_list, label = 'Model')
        amom[2].plot(xvec, sigma_y_list, label = 'Sigma y')
        amom[2].set_ylim([-0.1, set_range(d_2ndmom_list, fac = 0.5)[1]])

        amom[2].set_xlabel('Along-wind Position [m]')
        amom[1].set_ylabel('Cross-wind Position [m]')
        amom[2].set_ylabel('One-sided Width [m]')
        _ = [am.legend() for am in amom]

        xlims = a2d[0].get_xlim()
        ylims = a2d[0].get_ylim()

        a2d[0].plot(x + pso[0], y + pso[1], color = 'black')
        a2d[1].plot(x + pso[0], y + pso[1], color = 'white')
        a2d[0].plot(xsmo + pso[0], ysmo + pso[1], color = 'white')
        a2d[1].plot(xsmo + pso[0], ysmo + pso[1], color = 'black')
        if full_retrieval_line is not None:
            a2d[0].plot(xf + pso[0], yf + pso[1], '--', color = 'white')
            a2d[1].plot(xf + pso[0], yf + pso[1], '--', color = 'black')
        
        #x, y = rotate_vector(theta_deg, np.vstack((x_alongwind_list, (np.array(d_1stmom_cleaned_m) + d_2ndmom_smoothed_m)/gsd)))
        x, y = rotate_vector(theta_deg, np.vstack((x_alongwind_list, (np.array(d_1stmom_m) + d_2ndmom_smoothed_m)/gsd)))
        a2d[0].plot(x + pso[0], y + pso[1], color = 'black')
        a2d[1].plot(x + pso[0], y + pso[1], color = 'white')

        #x, y = rotate_vector(theta_deg, np.vstack((x_alongwind_list, (np.array(d_1stmom_cleaned_m) - d_2ndmom_smoothed_m)/gsd)))
        x, y = rotate_vector(theta_deg, np.vstack((x_alongwind_list, (np.array(d_1stmom_m) - d_2ndmom_smoothed_m)/gsd)))
        a2d[0].plot(x + pso[0], y + pso[1], color = 'black')
        a2d[1].plot(x + pso[0], y + pso[1], color = 'white')

        _ = [a.set_xlim(xlims) for a in a2d]
        _ = [a.set_ylim(ylims) for a in a2d]

        _ = [f.tight_layout() for f, _ in f_list]
        f2d.tight_layout()
        fmom.tight_layout()

        if output_plot_basename is not None:
            fname = output_plot_basename + f'/{plume_id}_slices_'
            for i in range((n_slices-1) // n_slices_per_figure + 1):
                f_list[i][0].savefig(fname + f'cross_plume_slice_{plot_name_tag}{i:02d}.png', dpi = 150)
                #f_list[i][0].savefig(fname + f'cross_plume_slice_{i:00d}.pgf', backend = 'pgf')
            f2d.savefig(fname + f'plume_reference{plot_name_tag}.png', dpi = 150)
            fmom.savefig(fname + f'moments{plot_name_tag}.png', dpi = 150)

            # Combine all the individual cross track slices into one big image
            test = Image.open(fname + f'cross_plume_slice_{plot_name_tag}00.png')
            new_im = Image.new('RGB', (test.size[0]*len(f_list), test.size[1]))
            for i in range((n_slices-1) // n_slices_per_figure + 1):
                cpimage = Image.open(fname + f'cross_plume_slice_{plot_name_tag}{i:02d}.png')
                new_im.paste(cpimage, (test.size[0]*i, 0))
            new_im.save(fname + f'cross_plume_slice{plot_name_tag}.png')
    
    return d_1stmom_smoothed_m, d_2ndmom_smoothed_m, x_alongwind_list[-1]*gsd
        
def rotate_vector(angle_deg, vector):
    angle = angle_deg * np.pi / 180
    R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
    return R @ vector

def filter_loop(y, sigma, n_loops = 3, n_zeros = 100, show_filter_plots = False, width_sigma_threshold = 1.5):
    # Prepend zeros to help avoid big jumps at the beginning of the plume
    y_0 = np.concatenate((np.array([0]*n_zeros), y))
    sigma_0 = np.concatenate((np.array([0]*n_zeros), sigma)) 

    if show_filter_plots:
        pl.plot(np.arange(len(y_0))- n_zeros, y_0, num = 'filt', color = 'black')
        pl.plot(np.arange(len(y_0))- n_zeros, y_0, num = 'filt_data', color = 'black')

    # Iteratively replace outliers with zero
    for n_loop in np.arange(n_loops):
        y_0s = scipy.signal.savgol_filter(y_0, 19, 2)
        y_0 = np.where(np.abs(y_0 - y_0s) > width_sigma_threshold*sigma_0, 0, y_0 )

        if show_filter_plots:
            pl.plot(np.arange(len(y_0))- n_zeros, y_0s, num = 'filt', label = n_loop)
            pl.plot(np.arange(len(y_0))- n_zeros, y_0, num = 'filt_data', label = n_loop)

    y_0s = scipy.signal.savgol_filter(y_0, 19, 2)

    return y_0s[n_zeros:]
