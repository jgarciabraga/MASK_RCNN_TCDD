import ogr
import osr
import os

if __name__ == '__main__':

    total = 9
    # you must change this path for your computer
    # if you have two different set of images put one in test and other in prediction
    # chang the path for prediction
    path = 'C:\\Users\\Projeto\\Desktop\\GUIT\\dataset\\trees\\test_result\\'
    output_file = 'response.shp'
    file_start_with = 'pred_image'
    file_end_with = '.shp'
    driver_name = 'ESRI Shapefile'
    geometry_type = ogr.wkbPolygon
    src = osr.SpatialReference()
    out_driver = ogr.GetDriverByName(driver_name)
    # you must change the ESPG code for your region
    src.ImportFromEPSG(32723)

    final_path = path + output_file
    print('Save in: ', final_path)

    if os.path.exists(final_path):
        print('Exists!!!! Deleting!!!')
        out_driver.DeleteDataSource(final_path)
    else:
        print('File Free')

    out_ds = out_driver.CreateDataSource(final_path)
    out_layer = out_ds.CreateLayer(final_path, src, geom_type=geometry_type)

    for i in range(total):
        name = file_start_with + str(i) + file_end_with
        print('Nome: ', name)
        path_shape = path + name
        print('Path: ', path_shape)
        ds = ogr.Open(path_shape)
        layer = ds.GetLayer()
        print(layer.GetFeatureCount())
        for feat in layer:
            out_feat = ogr.Feature(out_layer.GetLayerDefn())
            out_feat.SetGeometry(feat.GetGeometryRef().Clone())
            out_layer.CreateFeature(out_feat)
            out_feat = None
            out_layer.SyncToDisk()
        # end for
    # end for

    out_ds = None
    del out_ds

#end if main