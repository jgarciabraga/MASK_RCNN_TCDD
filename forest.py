import osr
import gdal
import matplotlib.pyplot as plt
import numpy as np
from osgeo import ogr
import random
import skimage


def readimagetif(image_path):
    """
    :param image_path:
    is the path for the image to be computed
    :return: array_final, image_tif
    array_final is the nupy  array from image
    image_tif is the tif image read from disk
    """
    # read the geotiff image and return the numpy array
    # with the correct shape
    image_tif = gdal.Open(image_path)
    array_tif_image = image_tif.ReadAsArray()
    bands = array_tif_image.shape[0]
    rows = array_tif_image.shape[1]
    cols = array_tif_image.shape[2]
    array_final = np.empty([rows, cols, bands], dtype=np.float32)

    for k in range(bands):
        array_final[:, :, k] = array_tif_image[k, :, :]

    return array_final, image_tif


def converttofloat(array_final):
    """
    :param array_final:
    numpy array from image tif
    :return:
    numpy array from image tif in float number format
    """
    bands = array_final.shape[2]

    for k in range(bands):
        maximum = np.amax(array_final[:, :, k])
        minimum = np.amin(array_final[:, :, k])
        array_final[:, :, k] = ((array_final[:, :, k] - minimum) / (maximum - minimum))

    return array_final


def readshapefile(path_shape_file):
    """
    :param path_shape_file: the path for the shape file
    :return:
    shape the real shape file read from disk
    layer a shape file can be formed by several layers, but the normal is just one
    layer is the first layer in the shape file
    """
    shape = ogr.Open(path_shape_file)
    layer = shape.GetLayer(0)

    return shape, layer


def worldtopixel(x, y, geo_transform):
    """
    :param x:
    :param y:
    :param geo_transform:
    :return:
    """
    xPixel = []
    yPixel = []
    totalPoints = len(x)
    xOrigin = geo_transform[0]
    pixelWidth = geo_transform[1]
    yOrigin = geo_transform[3]
    pixelHeight = -geo_transform[5]

    for k in range(totalPoints):
        xPixel.append(int((x[k] - xOrigin) / pixelWidth))
        yPixel.append(int((yOrigin - y[k]) / pixelHeight))
    # end for

    return xPixel, yPixel


def createpolygonlist(layer, tif_image):
    """
    :param layer:
    :param tif_image:
    :return:
    """
    lat_polygons = []
    lon_polygons = []
    x_polygons = []
    y_polygons = []
    transform = tif_image.GetGeoTransform()

    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        geometry = feature.GetGeometryRef()
        r = geometry.GetGeometryRef(0)
        all_x = []
        all_y = []
        for k in range(r.GetPointCount()):
            all_x.append(r.GetX(k))
            all_y.append(r.GetY(k))
        # end for
        lat_polygons.append(all_y)
        lon_polygons.append(all_x)
    # end for

    for k in range(len(lat_polygons)):
        y = lat_polygons[k]
        x = lon_polygons[k]
        x, y = worldtopixel(x, y, transform)
        x_polygons.append(x)
        y_polygons.append(y)
    # end for

    return lon_polygons, lat_polygons, x_polygons, y_polygons


def getrandompolygonpixels(x_polygons, y_polygons):
    """
    :param x_polygons:
    :param y_polygons:
    :return:
    """

    total_of_features = len(x_polygons)
    choice = random.randint(0, (total_of_features - 1))
    x = x_polygons[choice]
    y = y_polygons[choice]

    return x, y


def getspeceficpolygonpixels(x_polygons, y_polygons, number):

    x = x_polygons[number]
    y = y_polygons[number]

    return x, y


def getpolygonbounds(x, y):
    """
    :param x: x coordinates from polygon in pixels
    :param y: y coordinates from polygon in pixels
    both x,y are the return from getspecificpolygon
    :return: return bounds - is the polygon boundbox
    """
    max_x = np.amax(x)
    min_x = np.amin(x)
    max_y = np.amax(y)
    min_y = np.amin(y)
    bounds = [min_x, max_x, min_y, max_y]

    return bounds


def getthecrown(tif_array, bounds, x, y):
    """
    :param tif_array:
    :param bounds:
    :param x:
    :param y:
    :return:
    """
    crown = tif_array[bounds[2]:bounds[3], bounds[0]:bounds[1], :].copy()
    bands = crown.shape[2]
    x_new = x - bounds[0]
    y_new = y - bounds[2]
    aux = np.zeros((crown.shape[0], crown.shape[1]), dtype=np.float)
    rr, cc = skimage.draw.polygon(y_new, x_new)
    aux[rr, cc] = 1.0

    for j in range(aux.shape[0]):
        for i in range(aux.shape[1]):
            if aux[j, i] == 0.0:
                for k in range(bands):
                    crown[j, i, k] = -1.0
                # end for
            # end if
        # end for
    # end for

    return crown


def checkdistance(points, x_position, y_position, distance):
    """
    :param points:
    :param x_position:
    :param y_position:
    :param distance:
    :return:
    """
    check = False
    for point in points:
        dist = np.sqrt((x_position - point[1]) ** 2 + (y_position - point[0]) ** 2)
        if dist >= distance:
            check = True
        else:
            check = False
            break

    return check


def get20initialpoints(numbers_array, window, points, distance):
    """
    :param numbers_array:
    :param window:
    :param points:
    :param distance:
    :return:
    """
    check = True
    height = numbers_array.shape[0]
    width = numbers_array.shape[1]
    quarter_height = np.int(height / 4)
    quarter_width = np.int(width / 4)

    while check:
        y_position = random.randint(quarter_height, height - quarter_height)
        x_position = random.randint(quarter_width, width - quarter_width)
        if len(points) == 0:
            if (y_position - window > quarter_height) and (x_position - window > quarter_width):
                if (y_position + window < height - quarter_height) and (x_position + window < width - quarter_width):
                    point = (y_position, x_position)
                    check = False
                    break
                # end if
            # end if
        else:
            if checkdistance(points, x_position, y_position, distance):
                if (y_position - window > quarter_height) and (x_position - window > quarter_width):
                    if (y_position + window < height - quarter_height) and (
                            x_position + window < width - quarter_width):
                        aux = numbers_array[(y_position - window):(y_position + window),
                              (x_position - window):(x_position + window)]
                        if np.count_nonzero(aux) == 0:
                            point = (y_position, x_position)
                            check = False
                            break
                        else:
                            check = True
                    else:
                        check = True
                else:
                    check = True
            else:
                check = True
            # end if else
        # end if else
    # end while
    return point


def getpoint(numbers_array, window, points, distance):
    """
    :param numbers_array:
    :param window:
    :param points:
    :param distance:
    :return:
    """
    check = True
    limit_height = numbers_array.shape[0]
    limit_width = numbers_array.shape[1]
    while check:
        y_position = random.randint(0, limit_height)
        x_position = random.randint(0, limit_width)
        if len(points) == 0:
            if (y_position - window > 0) and (x_position - window > 0):
                if (y_position + window < limit_height) and (x_position + window < limit_width):
                    point = (y_position, x_position)
                    check = False
                    break
                else:
                    check = True
        else:
            if checkdistance(points, x_position, y_position, distance):
                if (y_position - window > 0) and (x_position - window > 0):
                    if (y_position + window < limit_height) and (x_position + window < limit_width):
                        aux = numbers_array[(y_position - window):(y_position + window),
                              (x_position - window):(x_position + window)]
                        if np.count_nonzero(aux) == 0:
                            point = (y_position, x_position)
                            check = False
                            break
                        else:
                            check = True
                    else:
                        check = True
                else:
                    check = True
            else:
                check = True
            # end if else
        # end if else
    # end while

    return point


def getvectorbounds(px, py, crown, background_array):
    """
    :param px:
    :param py:
    :param crown:
    :param background_array:
    :return:
    """
    mask_height = crown.shape[0]
    mask_width = crown.shape[1]

    center_y = np.int(np.floor(mask_height / 2))
    center_x = np.int(np.floor(mask_width / 2))

    height_brick = background_array.shape[0]
    width_brick = background_array.shape[1]

    # from brick
    sup_x = px - center_x
    sup_y = py - center_y

    # from brick
    inf_x = px + center_x
    inf_y = py + center_y

    if (sup_x >= 0 and sup_y >= 0) and (inf_x < width_brick and inf_y < height_brick):
        point_sup = (0, 0)
        point_inf = (mask_height, mask_width)
        begin_x = px - center_x
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 1')
    elif (sup_x < 0 and sup_y < 0) and (inf_x < width_brick and inf_y < height_brick):
        dx = (center_x - px) - 1
        dy = (center_y - py) - 1
        point_sup = (dy, dx)
        point_inf = (mask_height, mask_width)
        begin_x = 0
        begin_y = 0
        if begin_x < 0 or begin_y < 0:
            print('problem case 2')
    elif (sup_x < 0 and sup_y >= 0) and (inf_x < width_brick and inf_y < height_brick):
        dx = (center_x - px) - 1
        point_sup = (0, dx)
        point_inf = (mask_height, mask_width)
        begin_x = 0
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 3')
    elif (sup_x < 0 and sup_y >= 0) and (inf_x < width_brick and inf_y >= height_brick):
        dx = (center_x - px) - 1
        point_sup = (0, dx)
        dy = height_brick - py
        point_inf = (center_y + dy, mask_width)
        begin_x = 0
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 4')
    elif (sup_x >= 0 and sup_y < 0) and (inf_x < width_brick and inf_y < height_brick):
        dy = (center_y - py) - 1
        point_sup = (dy, 0)
        point_inf = (mask_height, mask_width)
        begin_x = px - center_x
        begin_y = 0
        if begin_x < 0 or begin_y < 0:
            print('problem case 5')
    elif (sup_x >= 0 and sup_y < 0) and (inf_x >= width_brick and inf_y < height_brick):
        dy = (center_y - py) - 1
        point_sup = (dy, 0)
        dx = width_brick - px
        point_inf = (mask_height, center_x + dx)
        begin_x = px - center_x
        begin_y = 0
        if begin_x < 0 or begin_y < 0:
            print('problem case 6')
    elif (sup_x >= 0 and sup_y >= 0) and (inf_x >= width_brick and inf_y < height_brick):
        point_sup = (0, 0)
        dx = width_brick - px
        point_inf = (mask_height, center_x + dx)
        begin_x = px - center_x
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 7')
    elif (sup_x >= 0 and sup_y >= 0) and (inf_x >= width_brick and inf_y >= height_brick):
        point_sup = (0, 0)
        dx = width_brick - px
        dy = height_brick - py
        point_inf = (center_y + dy, center_x + dx)
        begin_x = px - center_x
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 8')
    elif (sup_x >= 0 and sup_y >= 0) and (inf_x < width_brick and inf_y >= height_brick):
        point_sup = (0, 0)
        dy = (height_brick - py)
        point_inf = (center_y + dy, mask_width)
        begin_x = px - center_x
        begin_y = py - center_y
        if begin_x < 0 or begin_y < 0:
            print('problem case 9')

    return point_sup, point_inf, begin_x, begin_y


def putcrown(point_sup, point_inf, begin_x, begin_y, crown, background_array, numbers_array, cont):
    """
    :param point_sup:
    :param point_inf:
    :param begin_x:
    :param begin_y:
    :param crown:
    :param background_array:
    :param numbers_array:
    :param cont:
    :return:
    """
    bands = background_array.shape[2]
    j = 0
    i = 0
    for in_y in range(point_sup[0], point_inf[0]):
        for in_x in range(point_sup[1], point_inf[1]):
            if crown[in_y, in_x, 0] != -1:
                numbers_array[(begin_y + j), (begin_x + i)] = np.int16(cont)
                for k in range(bands):
                    background_array[(begin_y + j), (begin_x + i), k] = crown[in_y, in_x, k]
                # end for
            # end if
            i += 1
        # end for
        i = 0
        j += 1
    # end for
    return background_array, numbers_array


def getsmallcrown(layer, tif_image, tif_array, limits):
    """
    :param layer:
    :param tif_image:
    :param tif_array:
    :param limits:
    :return:
    """
    area = []
    idet = 0
    small = 0

    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        geometry = feature.GetGeometryRef()
        r = geometry.GetGeometryRef(0)
        if (r.GetArea() > limits[0]) and (r.GetArea() < limits[1]):
            small = r.GetArea()
            idet = i
    # end for

    all_x = []
    all_y = []

    feature = layer.GetFeature(idet)
    geometry = feature.GetGeometryRef()
    r = geometry.GetGeometryRef(0)

    for k in range(r.GetPointCount()):
        all_x.append(r.GetX(k))
        all_y.append(r.GetY(k))
    # end for
    transform = tif_image.GetGeoTransform()
    all_x, all_y = worldtopixel(all_x, all_y, transform)
    bounds = getpolygonbounds(all_x, all_y)
    small_crown = getthecrown(tif_array, bounds, all_x, all_y)

    return small_crown


def cheekforfreespace(numbers_array, small_crown, window_small):
    """

    :param numbers_array:
    :param small_crown:
    :return:
    """

    begin_j = np.int(np.floor(small_crown.shape[0] / 2))
    end_j = numbers_array.shape[0] - begin_j
    begin_i = np.int(np.floor(small_crown.shape[1] / 2))
    end_i = numbers_array.shape[1] - begin_i
    found = False
    total_of_nozero = 1
    point = (-1, -1)

    for j in range(begin_j, end_j, begin_j):
        for i in range(begin_i, end_i, begin_i):
            aux = numbers_array[j - window_small:j + window_small, i - window_small:i + window_small]
            total_of_nozero = np.count_nonzero(aux)
            if total_of_nozero == 0:
                point = (j, i)
                found = True
                break
            else:
                point = (-1, -1)
            # end if
        # end for
        if found:
            break
    # end for

    return point, found


def fillfreespace(background_array, numbers_array, point, crown):
    """
    :param background_array:
    :param numbers_array:
    :param point:
    :param crown:
    :return:
    """
    number = np.amax(numbers_array) + 1

    begin_y = np.int(point[0] - np.floor(crown.shape[0] / 2))
    begin_x = np.int(point[1] - np.floor(crown.shape[1] / 2))

    for j in range(crown.shape[0]):
        for i in range(crown.shape[1]):
            if crown[j, i, 0] != -1.0:
                numbers_array[begin_y + j, begin_x + i] = number
                for k in range(crown.shape[2]):
                    background_array[begin_y + j, begin_x + i, k] = crown[j, i, k]
                # end for
            # end if
        # end for
    # end for

    return background_array, numbers_array

def array2raster(raster_path, type_array, raster_origin, pixel_height, pixel_width, rot_y, rot_x, drive, projection,
                 array):
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
    control = False

    if type_array == 'Float':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Float64)
    elif type_array == 'Byte':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_Byte)
    elif type_array == 'Integer':
        out_raster = drive.Create(raster_path, cols, rows, bands, gdal.GDT_UInt16)
    else:
        print('Type is incorrect')
        return control

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

    control = True

    return control


def array2polygon(raster_path, shape_path, layer_name):
    """

    :param raster_path:
    :param shape_path:
    :param layer_name:
    :return:
    """
    control = False
    src_ds = gdal.Open(raster_path)
    band = src_ds.GetRasterBand(1)
    mask = band
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dst_ds = driver.CreateDataSource(shape_path)
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(src_ds.GetProjectionRef())
    layer = dst_ds.CreateLayer(layer_name, srs=spatial_ref)
    fd = ogr.FieldDefn('DN', ogr.OFSTInt16)
    layer.CreateField(fd)
    dst_field = 0
    gdal.Polygonize(band, mask, layer, dst_field, [], None)
    control = True

    return control


def increasearray(array, step):
    """
    :param array:
    :param step:
    :return:
    """

    bands = 0
    type_array = array.dtype
    if type_array == 'int16':
        if array.ndim == 2:
            bands = 1
            final_array = np.zeros((array.shape[0] + np.int((2 * step)), array.shape[1] + np.int((2 * step))),
                                   dtype=np.int16)
        else:
            bands = array.shape[2]
            final_array = np.zeros((array.shape[0] + np.int((2 * step)),
                                    array.shape[1] + np.int((2 * step), array.shape[2]), array.shape[2]),
                                   dtype=np.int16)
    elif type_array == 'float32':
        if array.ndim == 2:
            bands = 1
            final_array = np.zeros((array.shape[0] + np.int((2 * step)), array.shape[1] + np.int((2 * step))),
                                   dtype=np.float32)
        else:
            bands = array.shape[2]
            final_array = np.zeros(
                (array.shape[0] + np.int((2 * step)), array.shape[1] + np.int((2 * step)), array.shape[2]),
                dtype=np.float32)

    # copy to the final array
    if bands == 1:
        rows = final_array.shape[0]
        cols = final_array.shape[1]
        final_array[step:(rows - step), step:(cols - step)] = array
    else:
        rows = final_array.shape[0]
        cols = final_array.shape[1]
        for k in range(bands):
            final_array[step:(rows - step), step:(cols - step), k] = array[:, :, k]

    return final_array


def createsyntheticforest(tif_image, tif_array, layer, background_array, dist_tree, amnt_tree):
    """
    :param tif_image:
    :param tif_array:
    :param layer:
    :param background_array:
    :param dist_tree:
    :param amnt_tree:
    :return:
    """

    numbers_array = np.zeros((background_array.shape[0], background_array.shape[1]), dtype=np.int16)
    points = []
    lon, lat, x, y = createpolygonlist(layer, tif_image)
    # With uou want to develop your strategy to dill the synthetic images, go ahead
    # Comment the code between comment here and put you strategy
    # Bellow, I left some strategies developed by myself

    # I applied this code to create forest with more than 100 trees
    # comment here
    for k in range(amnt_tree):
        pol_x, pol_y = getrandompolygonpixels(x, y)
        bounds = getpolygonbounds(pol_x, pol_y)
        vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
        window = np.amax(vect)
        window = np.int(window / 2)

        if k == 0 and window < 20:
            while window < 20 or window > 32:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window / 2)
            # end while
        elif (k == 1 or k == 2 or k == 3 or k == 4) and window > 20:
            while window < 10 or window > 20:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window / 2)
            # end while
        elif (k > 4 and k < 70) and window > 5:
            while window > 5:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window / 2)
            # end while
        elif k >= 70 and window >= 3:
            while window >= 3:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window / 2)
            # end while
        # end else if
    # comment here

        '''
        # this code I applied to create synthetic images with a maximum of 50 trees
        # comment here
        if k == 0 and window < 20:
            while window < 20 or window > 30:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window/2)
            #end while
        elif k > 0 and k < 3:
            while window < 10 or window > 20:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window/2)
            #end while
        elif k > 3:
            while window > 8:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window/2)
            #end while
        #end if
        # comment here
        '''

        '''
        # I used this strategy also
        # comment here
        if k == 0 and window < 25:
            #print('entrei')
            teste = True
            while teste:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window/2)
                if window > 20 and window < 30:
                    teste = False
                else:
                    teste = True
            #print(window)
        elif (k > 0 and k < 15) and window >= 10:
            while window >= 10:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window/2)
            #end while
        elif k > 15 and window >= 5:
            while window >= 5:
                pol_x, pol_y = getrandompolygonpixels(x, y)
                bounds = getpolygonbounds(pol_x, pol_y)
                vect = [bounds[1] - bounds[0], bounds[3] - bounds[2]]
                window = np.amax(vect)
                window = np.int(window / 2 * 0.7)
            #end while
        #end if
        # comment here
        '''

        crown = getthecrown(tif_array, bounds, pol_x, pol_y)
        # If you want to fill the images first in the center and after near of image's edges
        # You should set up k to > 0
        # k is the number of trees that will fill the image's center
        if k == 0:
            point = get20initialpoints(numbers_array, window, points, dist_tree / 2)
        else:
            point = getpoint(numbers_array, window, points, dist_tree)
        point_sup, point_inf, begin_x, begin_y = getvectorbounds(point[1], point[0], crown, background_array)
        background_array, numbers_array = putcrown(point_sup, point_inf, begin_x, begin_y, crown, background_array,
                                                   numbers_array, k + 1)
        points.append(point)
    # end for

    return background_array, numbers_array


def completesyntheticforest(layer, tif_image, tif_array, background_array, numbers_array, limits):
    """

    :param layer:
    :param tif_image:
    :param tif_array:
    :param background_array:
    :param numbers_array:
    :param limits
    :return:
    """
    window_small = 2
    inf_limit = random.uniform(limits[0], limits[1])
    sup_limit = random.uniform(inf_limit, limits[1])
    small_crown = getsmallcrown(layer, tif_image, tif_array, (inf_limit, sup_limit))
    found = True
    while (found):
        point, found = cheekforfreespace(numbers_array, small_crown, window_small)
        if (found):
            background_array, numbers_array = fillfreespace(background_array, numbers_array, point, small_crown)
        # end if
    # end while

    return background_array, numbers_array


if __name__ == '__main__':

    # change this for your computer
    path_image = 'C:/Users/Projeto/Desktop/GUIT/santagenebra_examples.tif'
    # change this for your computer
    path_shape = 'C:/Users/Projeto/Desktop/GUIT/examples.shp'
    # change this for your computer
    path_background = 'C:/Users/Projeto/Desktop/GUIT/background.tif'
    # change this for your computer
    # change the train to validation to create images for validation
    raster_path = 'C:/Users/Projeto/Desktop/GUIT/dataset/trees/train/'
    shape_path = 'C:/Users/Projeto/Desktop/GUIT/dataset/trees/train/'
    temp_path = 'C:/Users/Projeto/Desktop/GUIT/dataset/trees/train/temp.tif'
    # the name of the image
    # do not change
    raster_name = 'image'
    # the name of the shape
    # do not change
    shape_name = 'shape'
    raster_ext = '.tif'
    shape_ext = '.shp'
    # distance between trees. You can put others values
    distance = 5
    # total of tree en each image
    total_tree = 150
    # total of images
    total_forest = 10
    # the images must start with 0, k = 0, and go on.
    # if tou create 10 images, k must start with 10 in the nest time
    # check the train or validation folder for this
    k = 0

    for i in range(total_forest):
        tif_array, tif_image = readimagetif(path_image)
        shape, layer = readshapefile(path_shape)
        background_array, background_tif = readimagetif(path_background)
        background_array, numbers_array = createsyntheticforest(tif_image, tif_array, layer, background_array, distance, total_tree)
        numbers_array = increasearray(numbers_array, 2)
        background_array = increasearray(background_array, 2)

        # create a raster from synthetic forest
        origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height = background_tif.GetGeoTransform()
        drv = background_tif.GetDriver()
        raster_path_final = raster_path + raster_name + str(k) + raster_ext
        raster_origin = (origin_y, origin_x)
        projection = background_tif.GetProjection()
        # you can change this for
        type_array = 'Integer'
        control = array2raster(raster_path_final, type_array, raster_origin, pixel_height, pixel_width, rot_y, rot_x, drv, projection, background_array)

        # create a temp raster
        origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height = background_tif.GetGeoTransform()
        drv = background_tif.GetDriver()
        temp_origin = (origin_y, origin_x)
        projection = tif_image.GetProjection()
        type_array = 'Integer'
        control = array2raster(temp_path, type_array, temp_origin, pixel_height, pixel_width, rot_y, rot_x, drv, projection, numbers_array)

        # create a shape from synthetic forest
        shape_path_final = shape_path + shape_name + str(k) + shape_ext
        layer_name = 'crowns'
        control = array2polygon(temp_path, shape_path_final, layer_name)
        print('Done: ', i+1)
        print('Forest ', k)
        k += 1

        # end for
    print('Finished!!!!')
