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
import jax
import jax.numpy as jnp

def point_gaussian_plume_model(nx, ny, dx, dy, x0_grid, y0_grid, pnt,
                                      plume_x_m, plume_y_m,
                                      plume_mean_locations_spacing_m,
                                      C_Q_kg_per_s_mpers, x0_param, a, exponent, theta_deg, plume_mean_locations,
                                      do_exp):
    '''2D Gaussian plume'''

    # Rotate coordinate grid
    tr = -1*theta_deg * jnp.pi / 180
    x = jnp.cos(tr)*x0_grid - jnp.sin(tr)*y0_grid
    y = jnp.sin(tr)*x0_grid + jnp.cos(tr)*y0_grid

    # Offset to plume location
    offx = jnp.cos(tr)*plume_x_m - jnp.sin(tr)*plume_y_m
    offy = jnp.sin(tr)*plume_x_m + jnp.cos(tr)*plume_y_m

    x = x - offx
    y = y - offy

    x = jnp.where(jnp.abs(x) < 0.01, 0.01, x)
    y = jnp.where(jnp.abs(y) < 0.01, 0.01, y)

    # Shift the plume in the cross plume direction (perpendicular to the plume angle theta_deg) according to plume_mean_locations
    if plume_mean_locations is None:
        plume_mean_locations_rot = 0.
    else:
        interp_locations = x.flatten()
        plume_mean_locations_rot = jnp.interp(jnp.array(interp_locations), 
                                              jnp.arange(len(plume_mean_locations))*plume_mean_locations_spacing_m,
                                              jnp.array(plume_mean_locations),
                                              left = 0, right = 0).reshape(y.shape)

    # Plume spreading model: empirical
    chi = jnp.abs(x) / x0_param
    a_clip = jnp.where(do_exp, a, 1.)
    aa = jnp.where(do_exp, jnp.exp(a_clip), jnp.array(a))

    sigma_y = aa * chi**exponent
    sigma_y2 = sigma_y**2

    const = 1 / jnp.sqrt(2 * jnp.pi * sigma_y2)
    expo = jnp.exp(-0.5 * (y - plume_mean_locations_rot)**2 / sigma_y2)

    v = jnp.where(do_exp, jnp.exp(C_Q_kg_per_s_mpers), C_Q_kg_per_s_mpers)
    plume_column_enhancement_kg_per_m2 = const * v * expo
    plume_column_enhancement_kg_per_m2 = jnp.where(x < 0, 0, plume_column_enhancement_kg_per_m2)

    # Convert to concentration length [ppm m] assuming standard temperature and pressure
    #     ppm(m)     L/m^3        mol/L      kg/mol
    k = (1.0/1e6)*(1000.0/1.0)*(1.0/22.4)*(0.01604/1.0) # scaling factor from ppm m to kg / m^2 CH4
    plume_column_enhancement_ppmm = plume_column_enhancement_kg_per_m2 / k

    return plume_column_enhancement_ppmm











#def Langevin_eddy_diffusion(x, y, plume_parameters_dict): '''This implements the Langevin spreading function as given in the "Langevin equation" section of the Eddy Diffusion Wikipedia page: https://en.wikipedia.org/wiki/Eddy_diffusion#Langevin_equation sigma_x**2(t) = 2 u**2 * TL (t - TL * (1 - exp(-t/TL))) In the implementation here, I assumed that t = x / wind_speed_ms. If that is inserted into the equation, then it becomes clear that the expression has only a single parameter: a = wind_speed_m_per_s * TL Values of the parameters that seem to give reasonable shaped plumes are choices of wind speed and TL that yield a approx. 0.01.  For example, wind_speed_m_per_s = 10, TL = 0.001.  Required plume_parameters_dict entries: 'wind_speed_m_per_s': float The wind speed in meters per second 'TL': float
#        Lagrangian integral time scale (see wikipedia page)
#    
#    Return:
#    -------
#    sigma_y2: ndarray of floats
#        The value of the Gaussian plume spreading function at each x, y position
#    '''
#    wind_speed_m_per_s = plume_parameters_dict['wind_speed_m_per_s']
#    TL = plume_parameters_dict['TL'] # Plume spreading parameter
#
#    a = wind_speed_m_per_s * TL
#
#    t = x / wind_speed_m_per_s
#
#    sigma_y2 = 2 * a * (x - a*(1 - np.exp(-1 * x / a)))
#
#    return sigma_y2
#