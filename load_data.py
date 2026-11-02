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

import subprocess
import os
import numpy as np
from osgeo import gdal
from osgeo import ogr
import json
import pandas as pd
from datetime import timedelta
import datetime
import pdb
import pickle
from rasterio.features import rasterize
from shapely.geometry import Polygon
import shapely
import shapely.ops
import glob
import pyproj
import gis_utils
import scipy.ndimage

import matplotlib.pyplot as plt

import spec_io

def get_orthoed_flags(scene_FIDs, flag_file_location):
    scene_unortho_glt_with_wildcards = [flag_file_location + '/' + scene_FID[4:12] + '/' + scene_FID + f'/l1b/{scene_FID}*_l1b_glt*.img' 
                                for scene_FID in scene_FIDs]

    scene_unortho_l1b_bandmask_with_wildcards = [flag_file_location + '/' + scene_FID[4:12] + '/' + scene_FID + f'/l1b/{scene_FID}*_l1b_bandmask*.img' 
                                                 for scene_FID in scene_FIDs]

    scene_unortho_l2a_mask_with_wildcards = [flag_file_location + '/' + scene_FID[4:12] + '/' + scene_FID + f'/l2a/{scene_FID}*_l2a_mask*.img' 
                                             for scene_FID in scene_FIDs]

    def get_with_glob(filename_with_wildcards):
        files = glob.glob(filename_with_wildcards)
        if len(files) != 1:
            raise IOError(f'Could not matching file with {filename_with_wildcards}')
        return files[0]

    scene_unortho_glt_filenames = []
    for scene_unortho_glt_with_wildcard in scene_unortho_glt_with_wildcards:
        scene_unortho_glt_filenames.append(get_with_glob(scene_unortho_glt_with_wildcard))
    
    scene_unortho_l1b_bandmask_filenames = []
    for scene_unortho_l1b_bandmask_with_wildcard in scene_unortho_l1b_bandmask_with_wildcards:
        scene_unortho_l1b_bandmask_filenames.append(get_with_glob(scene_unortho_l1b_bandmask_with_wildcard))
    
    scene_unortho_l2a_mask_filenames = []
    for scene_unortho_l2a_mask_with_wildcard in scene_unortho_l2a_mask_with_wildcards:
        scene_unortho_l2a_mask_filenames.append(get_with_glob(scene_unortho_l2a_mask_with_wildcard))
    
    scene_ortho_good_pixels_filenames = []
    for scene_unortho_glt_filename, scene_unortho_l1b_bandmask_filename, scene_unortho_l2a_mask_filename in \
            zip(scene_unortho_glt_filenames, scene_unortho_l1b_bandmask_filenames, scene_unortho_l2a_mask_filenames):
        
        m, _     = spec_io.load_data(scene_unortho_glt_filename)
        _, d_l1b = spec_io.load_data(scene_unortho_l1b_bandmask_filename)
        _, d_l2a = spec_io.load_data(scene_unortho_l2a_mask_filename)

        dilated_saturation_mask, _ = calculate_saturation_mask(d_l1b)
        clouds_and_surface_water_mask = np.sum(d_l2a[:,:,:3],axis=-1) > 0
        good_pixels = np.logical_not(np.logical_or(dilated_saturation_mask, clouds_and_surface_water_mask))

        good_pixels_ortho = spec_io.ortho_data(good_pixels, m.glt)

        s = os.path.basename(scene_unortho_l1b_bandmask_filename)
        scene_ortho_good_pixels_filename = os.path.join('/tmp', os.path.splitext(s)[0] + '_good_pixels_ORT.img')
        spec_io.write_envi_file(good_pixels_ortho[:,:,np.newaxis].astype(np.float32), m, scene_ortho_good_pixels_filename)
        scene_ortho_good_pixels_filenames.append(scene_ortho_good_pixels_filename)
    
    return scene_ortho_good_pixels_filenames

def get_EMIT_scenes_and_plume_coordinates(plume_id, 
                    plume_json_filename = '/store/brodrick/methane/ch4_plumedir/previous_manual_annotation_oneback.json', 
                    #MF_data_location = '/store/brodrick/methane/plume_scenes_run_20240328',
                    MF_data_location = '/store/emit/ops/data/acquisitions/',
                    flag_file_location = '/store/emit/ops/data/acquisitions'):

    if not os.path.exists(plume_json_filename):
        raise ValueError(f'The plume json file {plume_json_filename} does not exist.')
    j = json.load(open(plume_json_filename))

    ##############################################################################################
    # If using the visions_delivery json file:
    #plume_IDs = [x['properties']['Plume ID'] for x in j['features']]
    #plume_ind = plume_IDs.index(plume_id)
    #scene_FIDs = j['features'][plume_ind]['properties']['Scene FIDs']
    #plume_coordinates = j['features'][plume_ind]['geometry']['coordinates']
    ##############################################################################################

    ##############################################################################################
    # If using the previous_manual_annotation_oneback json file:
    plume_IDs = [x['properties']['Plume ID'] for x in j['features']]
    plume_ind = plume_IDs.index(plume_id)
    scene_FIDs = j['features'][plume_ind]['properties']['fids']
    plume_coordinates_lon_lat = j['features'][plume_ind]['geometry']['coordinates']
    ##############################################################################################

    if len(scene_FIDs) == 0:
        raise ValueError(f'Plume ID {plume_id} has no matching scenes')
    if len(scene_FIDs) > 2:
        raise NotImplementedError(f'Plume ID {plume_id} has covers more than two scenes')
    
    ##############################################################################################
    # Matched filter concentration lengths
    #scene_mf_filenames = [MF_data_location + '/' + scene_FID[4:12] + '/' + scene_FID + '_ch4_mf_ort' for scene_FID in scene_FIDs]
    scene_mf_filenames = []
    for scene_FID in scene_FIDs:
        srch = os.path.join(MF_data_location, scene_FID[4:12], scene_FID, 'ghg', 'ch4', '*_ortch4_*.tif')
        tmp = glob.glob(srch)
        if len(tmp) != 1:
            raise FileNotFoundError(f'Could not find MF file')
        scene_mf_filenames.append(tmp[0])
    mf_filename = f'/tmp/{plume_id}.vrt'
    my_vrt = gdal.BuildVRT(mf_filename, scene_mf_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Uncertainty
    #scene_unc_filenames = [MF_data_location + '/' + scene_FID[4:12] + '/' + scene_FID + '_ch4_uncert_ort' for scene_FID in scene_FIDs]
    scene_unc_filenames = []
    for scene_FID in scene_FIDs:
        srch = os.path.join(MF_data_location, scene_FID[4:12], scene_FID, 'ghg', 'ch4', '*_ortuncertch4_*.tif')
        tmp = glob.glob(srch)
        if len(tmp) != 1:
            raise FileNotFoundError(f'Could not find uncert file')
        scene_unc_filenames.append(tmp[0])
    unc_filename = f'/tmp/{plume_id}_unc.vrt'
    my_vrt = gdal.BuildVRT(unc_filename, scene_unc_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Sensitivity
    #scene_sens_filenames = [MF_data_location + '/' + scene_FID[4:12] + '/' + scene_FID + '_ch4_sens_ort' for scene_FID in scene_FIDs]
    scene_sens_filenames = []
    for scene_FID in scene_FIDs:
        srch = os.path.join(MF_data_location, scene_FID[4:12], scene_FID, 'ghg', 'ch4', '*_ortsensch4_*.tif')
        tmp = glob.glob(srch)
        if len(tmp) != 1:
            raise FileNotFoundError(f'Could not find sens file')
        scene_sens_filenames.append(tmp[0])
    sens_filename = f'/tmp/{plume_id}_sens.vrt'
    my_vrt = gdal.BuildVRT(sens_filename, scene_sens_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Flags
    scene_good_flags_filenames = get_orthoed_flags(scene_FIDs, flag_file_location)
    good_pixels_filename = f'/tmp/{plume_id}_good_pixels_ORT.vrt'
    my_vrt = gdal.BuildVRT(good_pixels_filename, scene_good_flags_filenames)
    my_vrt = None
    ##############################################################################################

    pseudo_origin = json.loads(j['features'][plume_ind]['properties']['Psuedo-Origin'])['coordinates']

    ##############################################################################################
    # Get the plume coordinates in the ORT image space
    plume_poly_lon_lat = shapely.geometry.Polygon(plume_coordinates_lon_lat[0])
    # Not sure how to interpret the plume coordinates if this len(plume_coordinates_lon_lat) != 1
    if len(plume_coordinates_lon_lat) != 1:
        pdb.set_trace()

    input_ds = gdal.Open(mf_filename)
    trans = input_ds.GetGeoTransform()
    def coord_trans(x, y):
        glt_xpx = int(round((x - trans[0])/ trans[1]))
        glt_ypx = int(round((y - trans[3])/ trans[5]))
        return glt_xpx, glt_ypx

    # Get the plume boundary in ORT image space
    plume_poly_arr = shapely.geometry.Polygon(shapely.ops.transform(coord_trans, plume_poly_lon_lat))
    ##############################################################################################

    pseudo_origin_y, pseudo_origin_x = coord_trans(pseudo_origin[0], pseudo_origin[1])

    FIDs = [s.split('/')[-1].split('_')[0] for s in scene_mf_filenames]
    FIDs_str = ' '.join(FIDs)
    scene_date = pd.to_datetime(datetime.datetime.strptime(FIDs[0][:12], 'emit%Y%m%d'))

    return FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, None, sens_filename, unc_filename, \
           good_pixels_filename

def get_AV3_Nov_Apr_scenes_and_plume_coordinates(fid, MF_data_locations = ['/scratch/jchapman/stage/JFMF/070924_update/'], 
                                            OBS_data_locations = ['/scratch/jchapman/stage/JFMF/070924_update/'],
                                            shape_locations = ['/scratch/colemanr/av3/controlled_release_23/plume_annotation/', '/scratch/colemanr/av3/controlled_release_24/plume_annotation/'],
                                            use_jay_pseudo_origin = True):
    shp_fid = fid[-6:]

    if os.path.exists(f'/tmp/{fid}'):
        files = glob.glob(f'/tmp/{fid}*')
        for file in files:
            os.remove(file)

    #############################################################################################################
    # Get MF data
    scene_mf_filename = ''
    for MF_data_location in MF_data_locations:
        f, fl = f'{MF_data_location}/{fid}_*_CH4_ORT', f'{MF_data_location}/{fid}_*ch4_mf_ort'
        scene_mf_filenames = glob.glob(f) + glob.glob(fl)
        if len(scene_mf_filenames) == 1:
            scene_mf_filename = scene_mf_filenames[0]

    if len(scene_mf_filename) == 0:
        raise ValueError(f'Should match 1 AVIRIS filename, but matched {len(scene_mf_filenames)} instead.')

    cmd = f'gdalwarp {scene_mf_filename} /tmp/{fid} -of ENVI -co PROFILE=BASELINE'
    subprocess.run(cmd, shell = True)
    #############################################################################################################

    #############################################################################################################
    # Get sensitivity data
    scene_sens_filename = ''
    for MF_data_location in MF_data_locations:
        f, fl = f'{MF_data_location}/{fid}_*_CH4_ORT', f'{MF_data_location}/{fid}_*ch4_mf_sensitivity_ort'
        scene_sens_filenames = glob.glob(f) + glob.glob(fl)
        if len(scene_sens_filenames) == 1:
            scene_sens_filename = scene_sens_filenames[0]

    if len(scene_sens_filename) == 0:
        raise ValueError(f'Should match 1 AVIRIS sensitivity filename, but matched {len(scene_sens_filenames)} instead.')

    cmd = f'gdalwarp {scene_sens_filename} /tmp/{fid}_sens -of ENVI -co PROFILE=BASELINE'
    subprocess.run(cmd, shell = True)
    #############################################################################################################

    #############################################################################################################
    # Get uncertainty data
    scene_unc_filename = ''
    for MF_data_location in MF_data_locations:
        f, fl = f'{MF_data_location}/{fid}_*_CH4_ORT', f'{MF_data_location}/{fid}_*ch4_mf_uncert_ort'
        scene_unc_filenames = glob.glob(f) + glob.glob(fl)
        if len(scene_unc_filenames) == 1:
            scene_unc_filename = scene_unc_filenames[0]

    if len(scene_unc_filename) == 0:
        raise ValueError(f'Should match 1 AVIRIS uncertainty filename, but matched {len(scene_unc_filenames)} instead.')

    cmd = f'gdalwarp {scene_unc_filename} /tmp/{fid}_unc -of ENVI -co PROFILE=BASELINE'
    subprocess.run(cmd, shell = True)
    #############################################################################################################

    ##############################################################################################
    # Get OBS data
    scene_obs_filename = ''
    for OBS_data_location in OBS_data_locations:
        f, fl = f'{OBS_data_location}/{fid}_*_OBS_ORT', f'{OBS_data_location}/{fid}_*obs_ort'
        scene_obs_filenames = glob.glob(f) + glob.glob(fl)
        if len(scene_obs_filenames) == 1:
            scene_obs_filename = scene_obs_filenames[0]

    if len(scene_obs_filename) == 0:
        raise ValueError(f'Should match 1 AVIRIS OBS filename, but matched {len(scene_obs_filenames)} instead.')

    if not os.path.exists(f'/tmp/{fid}_obs'):
        cmd = f'gdalwarp {scene_obs_filename} /tmp/{fid}_obs -b 10 -of ENVI -co PROFILE=BASELINE'
        subprocess.run(cmd, shell = True)
    
    #############################################################################################################

    #############################################################################################################
    # Get shape file data
    for shape_location in shape_locations:
        shape_filename = shape_location + f'id_{shp_fid}.shp'
        if os.path.exists(shape_filename):
            break

    o = ogr.Open(shape_filename)
    gl = o.GetLayer(0)
    gf = gl.GetFeature(0)
    j = gf.ExportToJson()
    shp_json = json.loads(j)

    plume_poly_in = shapely.geometry.shape(shp_json['geometry'])

    # Get the plume coordinates in the ORT image space
    input_ds = gdal.Open(scene_mf_filename)
    trans = input_ds.GetGeoTransform()
    def coord_trans(x, y):
        glt_xpx = int(round((x - trans[0])/ trans[1]))
        glt_ypx = int(round((y - trans[3])/ trans[5]))
        return glt_xpx, glt_ypx

    # Get the plume boundary in ORT image space
    plume_poly_arr = shapely.geometry.Polygon(shapely.ops.transform(coord_trans, plume_poly_in))
    #############################################################################################################

    #############################################################################################################
    # Get pseudo origin data
    data_proj = pyproj.CRS(input_ds.GetProjection())
    proj_wgs84 = pyproj.CRS('EPSG:4326')
    project_wgs84_to_data = pyproj.Transformer.from_crs(proj_wgs84, data_proj).transform
    pseudo_origin_data = shapely.ops.transform(project_wgs84_to_data, shapely.geometry.Point([33.963108, -117.336191]))
    pseudo_origin_x, pseudo_origin_y = [pseudo_origin_data.xy[0][0], pseudo_origin_data.xy[1][0]]
    if use_jay_pseudo_origin:
        df = pd.read_csv('/beegfs/scratch/jfahlen/av3_flight_info/av3_controlled_release_pseudo_origin.csv').set_index('plume_id')
        pseudo_origin_x, pseudo_origin_y = int(df.loc[fid]['pseudo_origin_x']), int(df.loc[fid]['pseudo_origin_y'])
    #############################################################################################################

    FIDs_str = fid
    scene_date = pd.to_datetime(datetime.datetime.strptime(fid[:11], 'AV3%Y%m%d'))

    return FIDs_str, scene_date, f'/tmp/{fid}', plume_poly_arr, pseudo_origin_x, pseudo_origin_y, f'/tmp/{fid}_obs', f'/tmp/{fid}_sens', f'/tmp/{fid}_unc'

def get_AVNG_Sept18_scenes_and_plume_coordinates(fid, MF_data_locations = ['/beegfs/scratch/jfahlen/ang_data/controlled_release_Sep2018'], 
                                            OBS_data_locations = None,
                                            shape_locations = None, 
                                            release_lat_lon = [34.756121, -117.276885],
                                            plume_metadata_filename = '/beegfs/scratch/jfahlen/ang_data/controlled_release_Sep2018/collection_metadata_geopandas.pickle',
                                            use_jay_pseudo_origin = False):
    shp_fid = fid[-6:]

    if os.path.exists(f'/tmp/{fid}'):
        files = glob.glob(f'/tmp/{fid}*')
        for file in files:
            os.remove(file)

    #############################################################################################################
    # Get MF data
    scene_mf_filename = ''
    for MF_data_location in MF_data_locations:
        f = f'{MF_data_location}/{fid}_ch4mf_v2t1_img'
        scene_mf_filenames = glob.glob(f)
        if len(scene_mf_filenames) == 1:
            scene_mf_filename = scene_mf_filenames[0]

    if len(scene_mf_filename) == 0:
        raise ValueError(f'Should match 1 AVNG filename, but matched {len(scene_mf_filenames)} instead.')

    temp_mf_filename = f'/tmp/{fid}_mf'
    if os.path.exists(temp_mf_filename):
        os.remove(temp_mf_filename)

    #cmd = f'gdal_translate -b 4 {scene_mf_filename} {temp_mf_filename}'
    cmd = f'gdalwarp {scene_mf_filename} {temp_mf_filename} -b 4 -of ENVI -co PROFILE=BASELINE'
    subprocess.run(cmd, shell = True)
    scene_mf_filename = temp_mf_filename
    
    
    #cmd = f'gdalwarp {scene_mf_filename} /tmp/{fid} -of ENVI -co PROFILE=BASELINE'
    #subprocess.run(cmd, shell = True)
    #############################################################################################################

    #############################################################################################################
    # Get shape file data
    #for shape_location in shape_locations:
    #    shape_filename = shape_location + f'id_{shp_fid}.shp'
    #    if os.path.exists(shape_filename):
    #        break

    #o = ogr.Open(shape_filename)
    #gl = o.GetLayer(0)
    #gf = gl.GetFeature(0)
    #j = gf.ExportToJson()
    #shp_json = json.loads(j)

    #plume_poly_in = shapely.geometry.shape(shp_json['geometry'])
    plume_poly_in = None
    #############################################################################################################

    gdf = pickle.load(open(plume_metadata_filename, 'rb'))
    plume_poly_in = gdf.loc[fid]['plume_boundary']
    plume_poly_in = shapely.ops.transform(lambda x, y: (y, x), plume_poly_in)
    pseudo_origin_ort_x = gdf.loc[fid]['pseudo_origin_x']
    pseudo_origin_ort_y = gdf.loc[fid]['pseudo_origin_y']
    # Get the plume coordinates in the ORT image space
    #input_ds = gdal.Open(scene_mf_filename)
    #trans = input_ds.GetGeoTransform()
    #def coord_trans(x, y):
    #    glt_xpx = int(round((x - trans[0])/ trans[1]))
    #    glt_ypx = int(round((y - trans[3])/ trans[5]))
    #    return glt_xpx, glt_ypx


    ##############################################################################################################
    ## Get pseudo origin data
    #data_ds = gdal.Open(scene_mf_filename)
    #data_proj = pyproj.CRS(data_ds.GetProjection())
    #proj_wgs84 = pyproj.CRS('EPSG:4326')
    #project_wgs84_to_data = pyproj.Transformer.from_crs(proj_wgs84, data_proj).transform
    #pseudo_origin_data = shapely.ops.transform(project_wgs84_to_data, shapely.geometry.Point([release_lat_lon[0], release_lat_lon[1]]))
    #pseudo_origin_ort_y, pseudo_origin_ort_x = coord_trans(pseudo_origin_data.xy[0][0], pseudo_origin_data.xy[1][0])
    ##############################################################################################################

    FIDs_str = fid
    scene_date = pd.to_datetime(datetime.datetime.strptime(fid[:11], 'ang%Y%m%d'))
    return FIDs_str, scene_date, scene_mf_filename, plume_poly_in, pseudo_origin_ort_x, pseudo_origin_ort_y, None, None, None

def get_AV3_DAAC_data_scenes_and_coordinates(plume_id,
                                             local_data_location = '/store/jfahlen/earthaccess_local_files/',
                                             plume_json_filename = '/store/jfahlen/earthaccess_local_files/AV3_plumes_list.json',
                                             gas = 'CH4'):
    
    j = json.load(open(plume_json_filename))    

    plume_IDs = [x['properties']['Plume ID'] for x in j['features']]
    suffixes = ['', '_' + gas, f'_{gas}_p000']
    plume_ind = -1
    for suffix in suffixes:
        if plume_id + suffix in plume_IDs:
            plume_ind = plume_IDs.index(plume_id + suffix)
    if plume_ind < 0:
        raise IndexError(f'Could not find plume {plume_id} in plume_IDs')

    scene_FIDs = j['features'][plume_ind]['properties']['fids']
    plume_coordinates_lon_lat = j['features'][plume_ind]['geometry']['coordinates']
    
    # Catch the case where the FID shows a range: AV320241104t180240_001_003
    if len(scene_FIDs) == 1:
        if len(scene_FIDs[0].split('_')) == 3:
            f, start, end = scene_FIDs[0].split('_')
            scene_FIDs = [f'{f}_{x:03d}' for x in range(int(start), int(end)+1)]

    scene_numbers = [x.strip().split('_')[1] for x in scene_FIDs]
    scene_numbers = sorted(scene_numbers)
    name = scene_FIDs[0].strip().split('_')[0] + f'_{scene_numbers[0]}_{scene_numbers[-1]}'

    def select_file(fid, file_tag):
        folder = glob.glob(os.path.join(local_data_location, f'{fid}*'))
        if len(folder) != 1:
            raise IOError
            #pdb.set_trace()
        matches = glob.glob(folder[0] + f'/{fid}*' + file_tag)
        if len(matches) != 1:
            raise IOError
            #pdb.set_trace()
        return matches[0]
    
    mf_filename = ''
    try:
        mf_filename = select_file(name, '_CH4_ORT.vrt')
        unc_filename = select_file(name, '_CH4_UNC_ORT.vrt')
        sns_filename = select_file(name, '_CH4_SNS_ORT.vrt')
    except:
        pass

    if mf_filename == '':

        ##############################################################################################
        # Matched filter concentration lengths
        mf_filenames = [select_file(scene_FID, '_CH4_ORT.tif') for scene_FID in scene_FIDs]
        mf_filename = f'/tmp/{plume_id}_mf.vrt'
        my_vrt = gdal.BuildVRT(mf_filename, mf_filenames)
        my_vrt = None
        ##############################################################################################

        ##############################################################################################
        # Uncertainty
        unc_filenames = [select_file(scene_FID, '*_CH4_UNC_ORT.tif') for scene_FID in scene_FIDs]
        unc_filename = f'/tmp/{plume_id}_unc.vrt'
        my_vrt = gdal.BuildVRT(unc_filename, unc_filenames)
        my_vrt = None
        ##############################################################################################

        ##############################################################################################
        # Sensitivity
        sns_filenames = [select_file(scene_FID, '*_CH4_SNS_ORT.tif') for scene_FID in scene_FIDs]
        sns_filename = f'/tmp/{plume_id}_sns.vrt'
        my_vrt = gdal.BuildVRT(sns_filename, sns_filenames)
        my_vrt = None
        ##############################################################################################

    ##############################################################################################
    # Flags
    #scene_good_flags_filenames = get_orthoed_flags(scene_FIDs, flag_file_location)
    good_pixels_filename = None
    #my_vrt = gdal.BuildVRT(good_pixels_filename, scene_good_flags_filenames)
    #my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Get the plume coordinates in the ORT image space
    #plume_coordinates_lat_lon = [[x[1],x[0]] for x in plume_coordinates_lon_lat[0]]
    plume_coordinates_lat_lon = [[x[1],x[0]] for x in plume_coordinates_lon_lat]
    #plume_poly_arr = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, plume_coordinates_lon_lat, lat_lon_or_lon_lat = 'lon_lat')
    plume_poly_arr = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, plume_coordinates_lat_lon)
    plume_poly = shapely.geometry.Polygon(plume_poly_arr)
    ##############################################################################################

    pseudo_origin = json.loads(j['features'][plume_ind]['properties']['Psuedo-Origin'])['coordinates']
    pseudo_origin = [pseudo_origin[1], pseudo_origin[0]]
    #pseudo_origin_y, pseudo_origin_x = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, [pseudo_origin], lat_lon_or_lon_lat = 'lon_lat')[0]
    pseudo_origin_y, pseudo_origin_x = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, [pseudo_origin])[0]

    #FIDs = ['_'.join(s.split('/')[-1].split('_')[:2]) for s in mf_filenames]
    FIDs = scene_FIDs #['_'.join(s.split('/')[-1].split('_')[:2]) for s in mf_filenames]
    FIDs_str = ' '.join(FIDs)
    scene_date = pd.to_datetime(datetime.datetime.strptime(FIDs[0][:11], 'AV3%Y%m%d'))

    return FIDs_str, scene_date, mf_filename, plume_poly, pseudo_origin_x, pseudo_origin_y, \
           None, sns_filename, unc_filename, good_pixels_filename

def get_AV3_DAAC_combined_data_scenes_and_coordinates(plume_id,
                                             local_data_location = '/store/jfahlen/earthaccess_local_files/',
                                             plume_json_filename = '/store/jfahlen/earthaccess_local_files/AV3_plumes_list.json',
                                             gas = 'CH4'):
    
    j = json.load(open(plume_json_filename))    

    plume_IDs = [x['properties']['Plume ID'] for x in j['features']]
    suffixes = ['', '_' + gas, f'_{gas}_p000']
    plume_ind = -1
    for suffix in suffixes:
        if plume_id + suffix in plume_IDs:
            plume_ind = plume_IDs.index(plume_id + suffix)
    if plume_ind < 0:
        raise IndexError(f'Could not find plume {plume_id} in plume_IDs')
    
    scene_FIDs = j['features'][plume_ind]['properties']['fids']
    plume_coordinates_lon_lat = j['features'][plume_ind]['geometry']['coordinates']
    
    ## Catch the case where the FID shows a range: AV320241104t180240_001_003
    #if len(scene_FIDs) == 1:
    #    if len(scene_FIDs[0].split('_')) == 3:
    #        f, start, end = scene_FIDs[0].split('_')
    #        scene_FIDs = [f'{f}_{x:03d}' for x in range(int(start), int(end)+1)]

    #scene_numbers = [x.strip().split('_')[1] for x in scene_FIDs]
    #scene_numbers = sorted(scene_numbers)
    #name = scene_FIDs[0].strip().split('_')[0] + f'_{scene_numbers[0]}_{scene_numbers[-1]}'

    #def select_file(fid, file_tag):
    #    folder = glob.glob(os.path.join(local_data_location, f'{fid}*'))
    #    if len(folder) != 1:
    #        raise IOError
    #        #pdb.set_trace()
    #    matches = glob.glob(folder[0] + f'/{fid}*' + file_tag)
    #    if len(matches) != 1:
    #        raise IOError
    #        #pdb.set_trace()
    #    return matches[0]
    
    #mf_filename = f'{local_data_location}/{plume_id}/{plume_id}_mf_ort.tif'
    #unc_filename = f'{local_data_location}/{plume_id}/{plume_id}_uncert_ort.tif'
    #sns_filename = f'{local_data_location}/{plume_id}/{plume_id}_sens_ort.tif'
    mf_filename = f'{local_data_location}/{scene_FIDs[0]}/{scene_FIDs[0]}_mf_ort.tif'
    unc_filename = f'{local_data_location}/{scene_FIDs[0]}/{scene_FIDs[0]}_uncert_ort.tif'
    sns_filename = f'{local_data_location}/{scene_FIDs[0]}/{scene_FIDs[0]}_sens_ort.tif'

    #try:
    #    mf_filename = select_file(name, '_CH4_ORT.vrt')
    #    unc_filename = select_file(name, '_CH4_UNC_ORT.vrt')
    #    sns_filename = select_file(name, '_CH4_SNS_ORT.vrt')
    #except:
    #    pass

    #if mf_filename == '':

    #    ##############################################################################################
    #    # Matched filter concentration lengths
    #    mf_filenames = [select_file(scene_FID, '_CH4_ORT.tif') for scene_FID in scene_FIDs]
    #    mf_filename = f'/tmp/{plume_id}_mf.vrt'
    #    my_vrt = gdal.BuildVRT(mf_filename, mf_filenames)
    #    my_vrt = None
    #    ##############################################################################################

    #    ##############################################################################################
    #    # Uncertainty
    #    unc_filenames = [select_file(scene_FID, '*_CH4_UNC_ORT.tif') for scene_FID in scene_FIDs]
    #    unc_filename = f'/tmp/{plume_id}_unc.vrt'
    #    my_vrt = gdal.BuildVRT(unc_filename, unc_filenames)
    #    my_vrt = None
    #    ##############################################################################################

    #    ##############################################################################################
    #    # Sensitivity
    #    sns_filenames = [select_file(scene_FID, '*_CH4_SNS_ORT.tif') for scene_FID in scene_FIDs]
    #    sns_filename = f'/tmp/{plume_id}_sns.vrt'
    #    my_vrt = gdal.BuildVRT(sns_filename, sns_filenames)
    #    my_vrt = None
    #    ##############################################################################################

    ##############################################################################################
    # Flags
    #scene_good_flags_filenames = get_orthoed_flags(scene_FIDs, flag_file_location)
    good_pixels_filename = None
    #my_vrt = gdal.BuildVRT(good_pixels_filename, scene_good_flags_filenames)
    #my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Get the plume coordinates in the ORT image space
    #plume_coordinates_lat_lon = [[x[1],x[0]] for x in plume_coordinates_lon_lat[0]]
    plume_coordinates_lat_lon = [[x[1],x[0]] for x in plume_coordinates_lon_lat]
    #plume_poly_arr = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, plume_coordinates_lon_lat, lat_lon_or_lon_lat = 'lon_lat')
    plume_poly_arr = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, plume_coordinates_lat_lon)
    plume_poly = shapely.geometry.Polygon(plume_poly_arr)
    ##############################################################################################

    pseudo_origin = json.loads(j['features'][plume_ind]['properties']['Psuedo-Origin'])['coordinates']
    pseudo_origin = [pseudo_origin[1], pseudo_origin[0]]
    #pseudo_origin_y, pseudo_origin_x = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, [pseudo_origin], lat_lon_or_lon_lat = 'lon_lat')[0]
    pseudo_origin_y, pseudo_origin_x = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, [pseudo_origin])[0]

    #FIDs = ['_'.join(s.split('/')[-1].split('_')[:2]) for s in mf_filenames]
    FIDs = scene_FIDs #['_'.join(s.split('/')[-1].split('_')[:2]) for s in mf_filenames]
    FIDs_str = ' '.join(FIDs)
    scene_date = pd.to_datetime(datetime.datetime.strptime(FIDs[0][:11], 'AV3%Y%m%d'))

    return FIDs_str, scene_date, mf_filename, plume_poly, pseudo_origin_x, pseudo_origin_y, \
           None, sns_filename, unc_filename, good_pixels_filename

def get_EMIT_DAAC_data_scenes_and_coordinates(plume_id,
                                              local_data_location = '/store/jfahlen/earthaccess_local_files/',
                                              plume_json_filename = '/store/jfahlen/earthaccess_local_files/AV3_plumes_list.json',
                                              gas = 'CH4'):
    
    j = json.load(open(plume_json_filename))    

    plume_IDs = [x['properties']['Plume ID'] for x in j['features']]
    suffixes = ['', '_' + gas, f'_{gas}_p000']
    plume_ind = -1
    for suffix in suffixes:
        if plume_id + suffix in plume_IDs:
            plume_ind = plume_IDs.index(plume_id + suffix)
    if plume_ind < 0:
        raise IndexError(f'Could not find plume {plume_id} in plume_IDs')

    scene_FIDs_json = j['features'][plume_ind]['properties']['fids']
    plume_coordinates_lon_lat = j['features'][plume_ind]['geometry']['coordinates']
    
    # Catch the case where the FID shows a range: EMIT20250603T173453_004_005
    # EMIT scenes have a different time for each scene in an orbit (AV3 has the same time for each
    # scene in a single flight line). We need to match on orbit number. So find the orbit number
    # for the first scene, then match each folder: 
    # EMIT_L2B_CH4ENH_002_20250603T173453_2515412_004 -> EMIT20250603T173453_004
    # EMIT_L2B_CH4ENH_002_20250603T173505_2515412_005 -> EMIT20250603T173505_005
    if len(scene_FIDs_json) == 1:
        if len(scene_FIDs_json[0].split('_')) == 3:
            f, start, end = scene_FIDs_json[0].split('_')

            first_folder = glob.glob(os.path.join(local_data_location, f'EMIT_*{f[4:]}*{start}'))[0]
            orbit_num = first_folder.split('_')[-2]
            folders = [glob.glob(os.path.join(local_data_location, f'EMIT_*{orbit_num}_{i:03d}'))[0] for i in range(int(start), int(end)+1)]

            scene_FIDs = []
            for folder in folders:
                dt, scene_num = os.path.basename(folder).split('_')[-3], os.path.basename(folder).split('_')[-1]
                scene_FIDs.append(f'EMIT{dt}_{scene_num}')
    else:
        scene_FIDs = scene_FIDs_json

    def select_EMIT_DAAC_file(fid, file_tag):
        # EMIT_L2B_CH4ENH_002_20250603T173453_2515412_004 -> fid: EMIT20250603T173453_004
        folder = glob.glob(os.path.join(local_data_location, f'EMIT*{fid[4:-4]}*{fid[-4:]}'))
        if len(folder) != 1:
            pdb.set_trace()
        matches = glob.glob(os.path.join(folder[0], f'{file_tag}'))
        if len(matches) != 1:
            pdb.set_trace()
        return matches[0]

    ##############################################################################################
    # Matched filter concentration lengths
    mf_filenames = [select_EMIT_DAAC_file(scene_FID, '*CH4ENH*.tif') for scene_FID in scene_FIDs]
    mf_filename = f'/tmp/{plume_id}_mf.vrt'
    my_vrt = gdal.BuildVRT(mf_filename, mf_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Uncertainty
    unc_filenames = [select_EMIT_DAAC_file(scene_FID, '*CH4UNCERT*.tif') for scene_FID in scene_FIDs]
    unc_filename = f'/tmp/{plume_id}_unc.vrt'
    my_vrt = gdal.BuildVRT(unc_filename, unc_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Sensitivity
    sns_filenames = [select_EMIT_DAAC_file(scene_FID, '*CH4SENS*.tif') for scene_FID in scene_FIDs]
    sns_filename = f'/tmp/{plume_id}_sns.vrt'
    my_vrt = gdal.BuildVRT(sns_filename, sns_filenames)
    my_vrt = None
    ##############################################################################################

    ##############################################################################################
    # Flags
    l2a_mask_filenames = [select_EMIT_DAAC_file(scene_FID, '*L2A_MASK*.tif') for scene_FID in scene_FIDs]
    l2a_mask_filename = f'/tmp/{plume_id}_l2a_mask.vrt'
    my_vrt = gdal.BuildVRT(l2a_mask_filename, l2a_mask_filenames)
    my_vrt = None
    m_l2a_mask, d_l2a_mask = spec_io.load_data(l2a_mask_filename)
    clouds_and_surface_water_mask = np.sum(d_l2a_mask[:,:,:3],axis=-1) > 0
    good_pixels = np.logical_not(clouds_and_surface_water_mask)
    good_pixels_filename = f'/tmp/{plume_id}_good_pixels.img'
    spec_io.write_envi_file(good_pixels[:,:,np.newaxis].astype(np.float32), m_l2a_mask, good_pixels_filename)
    ##############################################################################################

    ##############################################################################################
    # Get the plume coordinates in the ORT image space
    plume_poly_arr = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, plume_coordinates_lon_lat, lat_lon_or_lon_lat = 'lon_lat')
    plume_poly = shapely.geometry.Polygon(plume_poly_arr)
    ##############################################################################################

    pseudo_origin = json.loads(j['features'][plume_ind]['properties']['Psuedo-Origin'])['coordinates']
    pseudo_origin_y, pseudo_origin_x = gis_utils.transform_lat_lon_to_image_array_idx(mf_filename, [pseudo_origin], lat_lon_or_lon_lat = 'lat_lon')[0]

    FIDs = scene_FIDs
    FIDs_str = ' '.join(FIDs)
    scene_date = pd.to_datetime(datetime.datetime.strptime(FIDs[0][:12], 'EMIT%Y%m%d'))

    return FIDs_str, scene_date, mf_filename, plume_poly, pseudo_origin_x, pseudo_origin_y, \
           None, sns_filename, unc_filename, good_pixels_filename

def load_MF_from_plume_ID(plume_id, \
                          show_plots = False, use_jay_pseudo_origin = True,
                          save_data_dict = False,
                          use_saved_data_dict = False,
                          saved_data_dict_location = None,
                          plume_json_filename = None,
                          instrument = None,
                          input_data_location = None):
    '''For the specified plume_id, load the orthorectified concentration length file ('fid_ch4_mf_ort'), create the plume mask, and calculated a few values.

    Parameters
    ----------
    plume_id: string
        The full plume ID or AVIRIS file ID. Ex: 'CH4_PlumeComplex-495' or 'AV320240422t222212'
    show_plots: bool
        Set to True to display a few diagnostic plots
    
    Returns:
    --------
    This function returns a dict with the following elements:
        concentration_length_ort_ppmm: 2d ndarray
            The estimated concentration length (matched filter) in ppm m in the orthorectified space corresponding to plume_id
        plume_mask_ort: 2d ndarray of bool
            The mask with True values inside the plume dilineation
        plume_ort_x, plume_ort_y: float
            The delineated plume boundary vertices in orthorectified space
        pseudo_origin_ort_x, pseudo_origin_ort_y: float
            The plume's pseudo origin in orthorectified space
        center_of_mass_ort_x, center_of_mass_ort_y: float
            The plume's center of mask in orthorectified space
        plume_angle_ccw_from_x_ort_rad: float
            The angle between x axis and the ray from the plume pseudo origin to the center of mass, counterclockwise (ccw) in orthorectified
            space and in radians
    '''
    if use_saved_data_dict:
        dict_filename = saved_data_dict_location + f'/{plume_id}_data_dict.pickle'
        if os.path.exists(dict_filename):
            return pickle.load(open(dict_filename, 'rb'))
        else:
            print(f'Could not find data dict file {dict_filename}. Loading data from scratch instead.')

    if instrument is None:
        # AVIRIS 3
        good_pixels_filename = None
        if plume_id[:3] == 'AV3':
            aviris = True
            instrument_name = 'AV3'
            an_av3_type = True
            try:
                FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename = \
                    get_AV3_Nov_Apr_scenes_and_plume_coordinates(plume_id, use_jay_pseudo_origin = use_jay_pseudo_origin)
            except:
                an_av3_type = False

            if an_av3_type == False:
                if plume_json_filename is None:
                    FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                        good_pixels_filename = get_AV3_DAAC_data_scenes_and_coordinates(plume_id)
                else:
                    FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                        good_pixels_filename = get_AV3_DAAC_data_scenes_and_coordinates(plume_id, plume_json_filename=plume_json_filename)


        # EMIT DAAC
        elif plume_id[:4] == 'EMIT' and '_p0' in plume_id:
            emit = True
            instrument_name = 'EMIT'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_EMIT_DAAC_data_scenes_and_coordinates(plume_id, plume_json_filename=plume_json_filename)

        # Standard EMIT Plume
        elif plume_id[:3] == 'CH4':
            emit = True
            instrument_name = 'EMIT'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_EMIT_scenes_and_plume_coordinates(plume_id, plume_json_filename=plume_json_filename)

        # ANG AVIRIS Next Gen
        elif plume_id[:3] == 'ang':
            instrument_name = 'ANG'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename = \
                get_AVNG_Sept18_scenes_and_plume_coordinates(plume_id, use_jay_pseudo_origin = use_jay_pseudo_origin)
        else:
            raise NotImplementedError(f'Plume ID {plume_id} not recognized.')

    else:
        if instrument.lower() == 'emit':
            emit = True
            instrument_name = 'EMIT'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_EMIT_scenes_and_plume_coordinates(plume_id, plume_json_filename=plume_json_filename)
        elif instrument.lower() == 'emit_daac':
            emit = True
            instrument_name = 'EMIT'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_EMIT_DAAC_data_scenes_and_coordinates(plume_id, plume_json_filename=plume_json_filename)
        elif instrument.lower() == 'av3':
            instrument_name = 'AV3'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_AV3_DAAC_data_scenes_and_coordinates(plume_id, plume_json_filename=plume_json_filename, local_data_location=input_data_location)
        elif instrument.lower() == 'av3_combined':
            instrument_name = 'AV3'
            FIDs_str, scene_date, mf_filename, plume_poly_arr, pseudo_origin_x, pseudo_origin_y, obs_filename, sens_filename, unc_filename, \
                good_pixels_filename = get_AV3_DAAC_combined_data_scenes_and_coordinates(plume_id, plume_json_filename=plume_json_filename, local_data_location=input_data_location)
        else:
            raise NotImplementedError(f'Plume ID {plume_id} not recognized.')
        
    def load_data_with_gdal(filename):
        data = None
        if filename is not None:
            input_ds = gdal.Open(filename)
            data = input_ds.ReadAsArray().squeeze()
        return data

    data = load_data_with_gdal(mf_filename)
    obs_data = load_data_with_gdal(obs_filename)
    sens_data = load_data_with_gdal(sens_filename)
    unc_data = load_data_with_gdal(unc_filename)
    
    if good_pixels_filename == None:
        good_pixels = np.ones_like(data).astype(bool)
        good_pixels[data == -9999] = False
    else:
        good_pixels = load_data_with_gdal(good_pixels_filename)
        good_pixels = np.where(good_pixels > 0.5, True, False)

    # Get the plume pseudo origin coordinates, center of mass, and min/max bounds in ORT image space
    if plume_poly_arr is not None:
        plume_ort_y, plume_ort_x = plume_poly_arr.exterior.coords.xy
        center_of_mass_y, center_of_mass_x = [i[0] for i in plume_poly_arr.centroid.xy]
        min_mask_ort_y, min_mask_ort_x, max_mask_ort_y, max_mask_ort_x = [int(round(x)) for x in plume_poly_arr.bounds]

        # Get the angle from the pseudo origin to the center of mass
        dx, dy = center_of_mass_x - pseudo_origin_x, center_of_mass_y - pseudo_origin_y
        plume_angle_rad = np.arctan2(dy, dx)

        mask = rasterize(shapes=[plume_poly_arr], out_shape=data.shape) > 0.5
    else:
        plume_ort_y, plume_ort_x = None, None
        center_of_mass_y, center_of_mass_x = None, None
        min_mask_ort_y, min_mask_ort_x, max_mask_ort_y, max_mask_ort_x = None, None, None, None
        plume_angle_rad = None
        mask = None

    # Get the GSD
    proj_ds = gdal.Warp('', mf_filename, dstSRS='EPSG:3857', format='VRT') 
    transform_3857 = proj_ds.GetGeoTransform() 
    gsd_ort_m = transform_3857[1] 


    times_dict = {'scene_date': scene_date}
    if obs_data is not None:
        times = obs_data[:,:]
        times_in_plume = times[mask]
        times_dict = {
            'scene_date': scene_date,

            'scene_start_datetime_UTC': scene_date + datetime.timedelta(hours = np.min(times[times>0])),
            'scene_end_datetime_UTC':   scene_date + datetime.timedelta(hours = np.max(times[times>0])),

            'plume_start_datetime_UTC': scene_date + datetime.timedelta(hours = np.min(times_in_plume)),
            'plume_end_datetime_UTC':   scene_date + datetime.timedelta(hours = np.max(times_in_plume))}

    if show_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(data.T / sens_data.T, origin = 'lower', vmin=0, vmax=1500) # The transpose and 'lower' make it so x,y = 0,0 is lower left corner, x is horizontal, x is first dimension of array
        ax.plot(plume_ort_x, plume_ort_y, color = 'white')
        ax.plot([pseudo_origin_x], [pseudo_origin_y], 'o', color = 'red')
        ax.plot([center_of_mass_x], [center_of_mass_y], 'o', color = 'black')

        print(f'center of mass = {center_of_mass_x}, {center_of_mass_y}')
        print(f'plume angle = {plume_angle_rad * 180 / np.pi} deg')
    
    d = {
        'plume_id': plume_id,
        'FID': FIDs_str,
        'concentration_length_ort_ppmm': data,
        'sensitivity': sens_data,
        'uncertainty': unc_data,
        'plume_ort_x': np.array(plume_ort_x),
        'plume_ort_y': np.array(plume_ort_y),
        'pseudo_origin_ort_x': pseudo_origin_x,
        'pseudo_origin_ort_y': pseudo_origin_y,
        'center_of_mass_ort_x': center_of_mass_x,
        'center_of_mass_ort_y': center_of_mass_y,
        'min_mask_ort_x': min_mask_ort_x,
        'max_mask_ort_x': max_mask_ort_x,
        'min_mask_ort_y': min_mask_ort_y,
        'max_mask_ort_y': max_mask_ort_y,
        'center_of_mass_ort_y': center_of_mass_y,
        'plume_mask_ort': mask,
        'plume_angle_ccw_from_x_ort_rad': plume_angle_rad,
        'gsd_ort_m': gsd_ort_m,
        'wind_speed_m_per_s': None, # Legacy, wind is now done elsewhere
        'instrument_name': instrument_name,
        'good_pixels': good_pixels
    }
    try:
        j = json.load(open(plume_json_filename,'r'))
        plume_ids = [f['properties']['Plume ID'] for f in j['features']]
        plume_ind = plume_ids.index(plume_id)
        d['plume_time_UTC'] = j['features'][plume_ind]['properties']['Plume Time UTC']
    except:
        d['plume_time_UTC'] = ''
    for key in times_dict.keys():
        d[key] = times_dict[key]
    
    if save_data_dict:
        if os.path.exists(saved_data_dict_location):
            pickle.dump(d, open(saved_data_dict_location + f'/{plume_id}_data_dict.pickle', 'wb'))
        else:
            print(f'Could not find saved_data_dict_location {saved_data_dict_location}. Data dict not saved.')

    return d

def calculate_saturation_mask(l1b_bandmask: str, dilation_iterations=10, chunk_edges=None):
    '''l1b_bandmask marks static bad pixels and saturated pixels. The minimum subtraction below
    removes the contributions from static bad pixels, except in instances when the radiance
    has been otherwise flagged with bad values (-9999). The bad9999 mask identifies these and
    excludes them.'''

    # NOTE: This does not match the parallel_mf.py implementation as we don't have the radiance data handy.
    # This function will be incorrect sometimes (see commented out lines below)

    #bad9999 = np.any(radiance == -9999, axis = 1)
    l1b_bandmask_unpacked = np.unpackbits(l1b_bandmask, axis= -1)
    l1b_bandmask_summed = np.sum(l1b_bandmask_unpacked, axis = -1)
    max_vals = np.max(l1b_bandmask_summed, axis = 0)
    #min_vals = np.min( np.where(bad9999, max_vals, l1b_bandmask_summed), axis = 0)
    min_vals = np.min(l1b_bandmask_summed, axis = 0)
    saturation_mask = l1b_bandmask_summed - min_vals
    #saturation_mask[bad9999] = 0
    dilated_saturation_mask = scipy.ndimage.binary_dilation(saturation_mask != 0, iterations = dilation_iterations) < 1
    return np.logical_not(dilated_saturation_mask), saturation_mask != 0