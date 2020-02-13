# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:03:00 2019

@author: USUARIO
"""

import gdal
import ogr
import osr
import numpy as np
import sys

def readimagetif(image_path, str_array_type):
    """
    :param image_path:
    is the path for the image to be computed
    :param str_array_type:
    is the type of array Float, Integer, Byte
    :return: array, image_tif
    array is the nupy  array from image
    image_tif is the tif image read from disk
    """
    
    # read the geotiff image and return the numpy array
    # with the correct shape
    
    image_tif = gdal.Open(image_path)
    image_array = image_tif.ReadAsArray()
    bands = image_array.shape[0]
    rows = image_array.shape[1]
    cols = image_array.shape[2]
    if str_array_type == 'Float':
        array_type = np.float64
    elif str_array_type == 'Integer':
        array_type = np.uint16
    elif str_array_type == 'Byte':
        array_type = np.uint8
    else:
        print('Impossible run for this type')
        sys.exit()
    # end if

    array = np.empty([rows, cols, bands], dtype=array_type)
    for k in range(bands):
        array[:, :, k] = image_array[k, :, :]
		
    return array, image_tif

def array2raster(raster_path, array_type, raster_origin, pixel_height, pixel_width, rot_y, rot_x, drive, projection, array):
    """
    :param raster_path:
    :param type_array:
    :param raster_origin:
    :param pixel_height:
    :param pixel_width:
    :param rot_y:
    :param rot_x:
    :param drive:
    :param projection:
    :param array:
    :return:
    """
    rows = array.shape[0]
    cols = array.shape[1]

    if array.ndim == 2:
        bands = 1
    else:
        bands = array.shape[2]

    origin_y = raster_origin[0]
    origin_x = raster_origin[1]

    if array_type == 'Float':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Float64)
    elif array_type == 'Byte':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Byte)
    elif array_type == 'Integer':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Int16)
    else:
        print('Type is incorrect')
        sys.exit()

    out_raster.SetGeoTransform((origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height))

    for i in range(bands):
        out_band = out_raster.GetRasterBand(i + 1)
        if bands == 1:
            out_band.WriteArray(array)
        else:
            out_band.WriteArray(array[:, :, i])
    # end for

    out_raster_srs = osr.SpatialReference(wkt=projection)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())
    out_band.FlushCache()


    return 'Raster created!!!!'

def raster2polygon(raster_path, shape_path, layer_name):
    """
    :param raster_path:
    :param shape_path:
    :param layer_name:
    :return:
    """
    
    src_ds = gdal.Open(raster_path)
    band = src_ds.GetRasterBand(1)
    mask = band
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(shape_path)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(src_ds.GetProjectionRef())
    layer = dst_ds.CreateLayer(layer_name, srs=spatial_ref)
    fd = ogr.FieldDefn('ID', ogr.OFSTInt16)
    layer.CreateField(fd)
    dst_field = 0
    gdal.Polygonize(band, mask, layer, dst_field, [], None)

    return 'Shapefile created!!!!'

def changecellvaluebythreshhold(array, thresh_hold, value):
    """
    """
    
    height = array.shape[0]
    width = array.shape[1]
    array_numbers = np.zeros((height, width), dtype=np.byte)
    cont = 0
    if array.ndim == 2:
        bands = 1
    else:
        bands = array.shape[2]
        
    if bands==1:
        for j in range(height):
            for i in range(width):
                if array[j,i] <= thresh_hold:
                    array[j,i] = value
                    array_numbers[j,i] = 1
                    print('Changing!!!')
                # end if
            # end if
        # end if
    else:
        for j in range(height):
            for i in range(width):
                if (array[j,i,0] <= thresh_hold[0]) and (array[j,i,1] <= thresh_hold[1]) and (array[j,i,2] <= thresh_hold[2]):
                    array[j,i,0] = value[0]
                    array[j,i,1] = value[1]
                    array[j,i,2] = value[2]
                    array_numbers[j,i] = 1
                    print(cont)
                    cont += 1
                else:
                    cont += 1
                # end if
            # end for
        # end for
    # end else
        
    return array, array_numbers

def getpolygonsmaskrcnn(shape_path, image_tif):
    """
    :param path_shape_file:
    :param image_tif:
    :return:
    """
    polygons = []
    transform = image_tif.GetGeoTransform()
    shape = ogr.Open(shape_path)
    layer = shape.GetLayer(0)
    for feat in range(layer.GetFeatureCount()):
        geometry = layer.GetFeature(feat)
        aux = geometry.GetGeometryRef()
        r = aux.GetGeometryRef(0)
        all_x = []
        all_y = []
        polygon = {'name': 'polygon', 'all_points_x': [], 'all_points_y': []}
        for k in range(r.GetPointCount()):
            all_x.append(r.GetX(k))
            all_y.append(r.GetY(k))
        # end for
        x_final, y_final = worldtopixel(all_x, all_y, transform)
        polygon['all_points_x'] = x_final
        polygon['all_points_y'] = y_final
        polygons.append(polygon)
    # end for
    return polygons

def worldtopixel(x, y, geotransform):
    """
    :param x:
    :param y:
    :param geotransform:
    :return:
    """
    x_pixel = []
    y_pixel = []
    total_points = len(x)
    x_origin = geotransform[0]
    pixel_width = geotransform[1]
    y_origin = geotransform[3]
    pixel_height = -geotransform[5]
    for k in range(total_points):
        x_pixel.append(int((x[k] - x_origin) / pixel_width))
        y_pixel.append(int((y_origin - y[k]) / pixel_height))
    return x_pixel, y_pixel


def majorityfilterforzero(array, filterdim):
    
    height = array.shape[0]
    width = array.shape[1]
    begin_y = filterdim[0] - 1
    begin_x = filterdim[1] - 1 
    end_y = height - filterdim[0]
    end_x = width - filterdim[1]
    
    