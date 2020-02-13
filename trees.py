# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:42:58 2019
@author: José Renato Garcia Braga
"""
import os
import datetime
import numpy as np
import skimage.draw
import gdalbasics as gdb
from imgaug import augmenters as iaa

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils

# you must change this for your path. Do not forget!!!!
pathRootDir = 'C:\\Users\\Projeto\\Desktop\\GUIT'
#
ROOT_DIR = os.path.abspath(pathRootDir)
COCO_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')


class TreesConfig(Config):

    NUM_CLASSES = 1 + 1
    NAME = 'trees'
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    EPOCHS = 0
    STEPS_PER_EPOCH = 4171
    VALIDATION_STEPS = 1001
    BACKBONE = "resnet50"
    DETECTION_MIN_CONFIDENCE = 0.5
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    USE_MINI_MASK = False
    IMAGE_CHANNEL_COUNT = 3
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    # I left this equal to 220, but during the training my images had a maximum of 150 tree crowns
    MAX_GT_INSTANCES = 220
    DETECTION_MAX_INSTANCES = 220
    TRAIN_ROIS_PER_IMAGE = 220
    RPN_TRAIN_ANCHORS_PER_IMAGE = 220
    #
    RPN_NMS_THRESHOLD = 0.9
    MEAN_PIXEL = np.array([105, 236, 189])
    DETECTION_NMS_THRESHOLD = 0.3
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
# end class trees

class TreesDataset(utils.Dataset):
    
    def load_trees(self, dataset_dir, subset, amnt_images):
        """ 
        dataset_dir is the dataset path
        subset can be train, val or prediction
        amnt_images the amount of images for train or the others subset
        """
        
        image_name = 'image'
        image_ext = '.tif'
        shape_name = 'shape'
        shape_ext = '.shp'
        total_images = int(amnt_images)
        str_array_type = 'Byte'
        
        # add a class
        # main dataset folder, class id, target name
        self.add_class('trees', 1, 'trees')
        assert subset in ['train', 'val', 'prediction', 'test']
        dataset_dir = os.path.join(dataset_dir, subset)
        
        for total in range(total_images):
            image_temp_name = image_name + str(total) + image_ext
            image_path = os.path.join(dataset_dir, image_temp_name)
            array, image_tif = gdb.readimagetif(image_path, str_array_type)
            shape_temp_name = shape_name + str(total) + shape_ext
            shape_path = os.path.join(dataset_dir, shape_temp_name)
            polygons = gdb.getpolygonsmaskrcnn(shape_path, image_tif)
            height = array.shape[0]
            width = array.shape[1]
            if array.ndim == 2:
                bands = 1
            else:
                bands = array.shape[2]
            # end if
            self.add_image('trees', image_id = image_temp_name, path = image_path, width = width, height = height, bands = bands, polygons = polygons)
        # end for
    # end load_trees
    
    def load_mask(self, image_id):
        
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # end for
        
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    #end load_mask
    
    def image_reference(self, image_id):
        
        info = self.image_info[image_id]
        if info['source'] == 'trees':
            return info['path']
        else:
            super(self.__class__, self).image_reference(image_id)
        # end if
    # end image_reference
    
def train(model):
    
    # Training dataset.
    train_dataset = TreesDataset()
    train_dataset.load_trees(args.dataset, 'train', args.train_images)
    train_dataset.prepare()

    # Validation dataset
    val_dataset = TreesDataset()
    val_dataset.load_trees(args.dataset, 'val', args.val_images)
    val_dataset.prepare()

    augmentation = iaa.SomeOf((0, 2), [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.OneOf([iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]), iaa.Multiply((0.5, 1.5))])

    # You can first train the heads
    print('Training Heads...')
    model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=30, augmentation=augmentation, layers='heads')

    # and after, You can train all the neural network
    # see parameters.txt to get hints

    #print('Training All One...')
    #model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE, epochs=40, augmentation=augmentation, layers='all')

    #print('Training All Two...')
    #model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/10, epochs=40, augmentation=augmentation, layers='all')

    #print('Training All Tree...')
    #model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/100, epochs=40, augmentation=augmentation, layers='all')

    #print('Training All Four...')
    #model.train(train_dataset, val_dataset, learning_rate=config.LEARNING_RATE/1000, epochs=40, augmentation=augmentation, layers='all')

    return 'Training Finished!!!!'


def prediction(model):
    shape_path_aux = args.dataset + '/result/'
    temp_path_prediction = args.dataset + '/result/' + 'TEMP_RESP.tif'
    shape_prediction_name = 'pred_image'
    shape_prediction_ext = '.shp'
    dataset_prediction = TreesDataset()
    dataset_prediction.load_trees(args.dataset, 'prediction', args.pred_images)
    dataset_prediction.prepare()
    total_images = int(args.pred_images)

    if len(dataset_prediction.image_info) == total_images:
        for k in range(total_images):
            path = dataset_prediction.image_info[k]['path']
            array, image_tif = gdb.readimagetif(path, 'Integer')
            result = model.detect([array], verbose=1)[0]
            masks = result['masks']
            dim_masks = masks.shape
            masks_final = np.zeros((dim_masks[0], dim_masks[1]), dtype=np.int32)
            for cont_mask in range(dim_masks[2]):
                aux = masks[:, :, cont_mask].copy()
                for j in range(aux.shape[0]):
                    for i in range(aux.shape[1]):
                        if aux[j, i] != 0:
                            masks_final[j, i] = (cont_mask + 1)
                        # end if
                    # end for
                # end for
            # end for

            # save the shape
            origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height = image_tif.GetGeoTransform()
            drive = image_tif.GetDriver()
            projection = image_tif.GetProjection()
            raster_origin = (origin_y, origin_x)
            gdb.array2raster(temp_path_prediction, 'Integer', raster_origin, pixel_height, pixel_width, rot_y, rot_x, drive, projection, masks_final)
            shape_path = shape_path_aux + shape_prediction_name + str(k) + shape_prediction_ext
            gdb.raster2polygon(temp_path_prediction, shape_path, 'result')
            masks_final = None
        # end for
    else:
        print('Incorrect number of prediction images!!!!!')

    print('Prediction Done')


def test(model):

    shape_path_aux = args.dataset + '/test_result/'
    temp_path_test = args.dataset + '/test_result/' + 'TEMP_RESP.tif'
    shape_test_name = 'pred_image'
    shape_test_ext = '.shp'
    dataset_test = TreesDataset()
    dataset_test.load_trees(args.dataset, 'test', args.test_images)
    dataset_test.prepare()
    total_images = int(args.test_images)

    if len(dataset_test.image_info) == total_images:
        for k in range(total_images):
            path = dataset_test.image_info[k]['path']
            array, image_tif = gdb.readimagetif(path, 'Integer')
            result = model.detect([array], verbose=1)[0]
            masks = result['masks']
            dim_masks = masks.shape
            masks_final = np.zeros((dim_masks[0], dim_masks[1]), dtype=np.int32)
            for cont_mask in range(dim_masks[2]):
                aux = masks[:, :, cont_mask].copy()
                for j in range(aux.shape[0]):
                    for i in range(aux.shape[1]):
                        if aux[j, i] != 0:
                            masks_final[j, i] = (cont_mask + 1)
                        # end if
                    # end for
                # end for
            # end for

            # save the shape
            origin_x, pixel_width, rot_x, origin_y, rot_y, pixel_height = image_tif.GetGeoTransform()
            drive = image_tif.GetDriver()
            projection = image_tif.GetProjection()
            raster_origin = (origin_y, origin_x)
            gdb.array2raster(temp_path_test, 'Integer', raster_origin, pixel_height, pixel_width, rot_y, rot_x, drive, projection, masks_final)
            shape_path = shape_path_aux + shape_test_name + str(k) + shape_test_ext
            gdb.raster2polygon(temp_path_test, shape_path, 'test')
            masks_final = None
        # end for
    else:
        print('Incorrect number of prediction images!!!!!')

    print('Test Done')


if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Train Mask R-CNN to detect Trees.')
    parser.add_argument('command', metavar='<command>', help="'train', or 'prediction'")
    parser.add_argument('--dataset', required=True, metavar="/path/to/tree/dataset/", help= 'Directory of the Trees dataset')
    parser.add_argument('--weights', required=True, metavar="/path/to/weights.h5", help= "Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--train_images', required=False, metavar="amount of train images", help="please, inform the amount of train images")
    parser.add_argument('--val_images', required=False, metavar="amount of  validation images", help="please, inform the amount of validation images")
    parser.add_argument('--pred_images', required=False, metavar="amount of prediction images", help="please, inform the amount of prediction images")
    parser.add_argument('--test_images', required=False, metavar="amount of prediction images", help="please, inform the amount of prediction images")

    args = parser.parse_args()
    
    if args.command == 'train':
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == 'prediction':
        assert args.dataset, "Argument --dataset is required for prediction"
    elif args.command == 'test':
        assert args.dataset, "Argument --dataset is required for test"
    else:
        assert args.dataset, "Argument --dataset is required"
    
    print('Weights: ', args.weights)
    print('Dataset: ', args.dataset)
    print('Logs: the weights are stored at', args.logs)
    
    if args.command == 'train':
        assert args.train_images, 'Argument --train_images is required for training'

    print('Total of Train Images: ', args.train_images)

    if args.command == 'train':
        assert args.val_images, 'Argument --val_images is required for training'

    print('Total of Validation Images: ', args.val_images)

    if args.command == 'prediction':
        assert args.pred_images, 'Argument --pred_images is required for training'

    print('Total of prediction images: ', args.pred_images)

    if args.command == 'test':
        assert args.test_images, 'Argument --test_images is required for training'

    print('Total of test images: ', args.test_images)

    if args.command == "train":
        config = TreesConfig()
    else:
        class InferenceConfig(TreesConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            # I kept the same configuration of the training
            NUM_CLASSES = 1 + 1
            NAME = 'trees'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            BACKBONE = "resnet50"
            DETECTION_MIN_CONFIDENCE = 0.5
            BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
            USE_MINI_MASK = False
            IMAGE_CHANNEL_COUNT = 3
            RPN_NMS_THRESHOLD = 0.9
            MEAN_PIXEL = np.array([105, 236, 189])
            DETECTION_NMS_THRESHOLD = 0.3
            IMAGE_RESIZE_MODE = "square"
            IMAGE_MIN_DIM = 128
            IMAGE_MAX_DIM = 128
            MAX_GT_INSTANCES = 220
            DETECTION_MAX_INSTANCES = 220
            TRAIN_ROIS_PER_IMAGE = 220
            RPN_TRAIN_ANCHORS_PER_IMAGE = 220
        # end InferenceConfig()
        config = InferenceConfig()
    # end else
    config.display()
    
    # Create model
    if args.command == 'train':
        model = modellib.MaskRCNN(mode='training', config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode='inference', config=config, model_dir=args.logs)

    print("Loading weights ", args.weights)
    if args.weights == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(COCO_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(args.weights, by_name=True)
    
    # Train or evaluate
    if args.command == 'train':
        train(model)
    elif args.command == 'prediction':
        prediction(model)
    elif args.command == 'test':
        test(model)
    else:
        print("'{}' this is not recognized. Use 'train' or prediction'".format(args.command))
