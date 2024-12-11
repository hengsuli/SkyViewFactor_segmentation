import numpy as np
from PIL import Image
import cv2
import glob
import os
from math import ceil
import matplotlib.pyplot as plt
import random
import shutil
import json

def get_classes(json_files):
    """ identify distinct classes in the labelled json file
    Args:
        json_files (list of str): list of filepaths to json files that contain labelling info 
    Returns:
        list of str: list of distinct classes labelled in the json files
    """
    classes_list = []
    for f in json_files:
        with open(f,'r') as file:
            data = json.load(file)
            classes = list(set([s['label'] for s in data['shapes']]))
            for c in classes:
                classes_list.append(c)
    
    return list(set(classes_list))

def prepare_annotated_data(json_files,dir_name = "data_annotated"):
    """ 
    create a folder called data_annotated which contains the image and json file
    Args:
        json_files (list of str): list of filepaths to json files that contain labelling info 
    """
    for f in json_files:
        with open(f,'r') as file:
            data = json.load(file)
        json_fp = os.path.basename(f)
        img_fp = data['imagePath']
        save_dir = os.path.join(os.getcwd(),dir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        img_dir = os.path.join(os.getcwd(),"SVF","SVF_photos")
        json_dir = os.path.join(os.getcwd(),"SVF","json_labelled_files")
        # copy json file
        shutil.copyfile(os.path.join(json_dir,json_fp),os.path.join(save_dir,json_fp))
        # copy JPEG file
        shutil.copyfile(os.path.join(img_dir,img_fp),os.path.join(save_dir,img_fp))
    return

def image_to_array(fp):
    """ returns image in numpy array given filepath
    Args:
        fp (str): file path of image
    Returns:
        np.ndarray: numpy array of the image
    """
    return np.asarray(Image.open(fp))

def circular_mask(img, save_dir=None):
    """ creates a circular mask for future masking purposes
    create circular mask for image
    Args:
        img (np.ndarray): image that has the same dimension as the SVF photo
        save_dir (str): directory where to save the circular_mask image. Exports an RGB binary mask (mxnxc) if save_dir is True
    """
    nrow, ncol = img.shape[0], img.shape[1]
    radius = nrow//2
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (ncol//2,nrow//2), radius, (1,1,1), -1)

    fig, axes = plt.subplots(1,3,figsize=(12,5))
    axes[0].imshow(img)
    axes[1].imshow(mask,vmin=0,vmax=1)
    img_copy = img.copy()
    # img_copy[img_copy>1] = 255
    axes[2].imshow(img_copy*mask)
    plt.show()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        # save mask
        img_save = Image.fromarray(mask)
        img_save.save(os.path.join(save_dir,"SVF_mask.png"))
    return

def crop_square(fp, resize=(1024,1024),plot = True):
    """ crops the SVF photo and resize it to 1024 x 1024px
    crops just the circular SVF photo
    Args: 
        fp: (str or np.ndarray) input can be a file path or a np.ndarray
    Returns:
        np.ndarray: returns cropped and resized image.
    """
    if isinstance(fp,str):
        img = image_to_array(fp)
    else:
        img = fp
    nrow, ncol = img.shape[0], img.shape[1]
    c_x, c_y = ncol//2,nrow//2
    radius = nrow//2
    cropped_img = img[:,c_x - radius: c_x + radius]
    if resize is not None:
        cropped_img = cv2.resize(cropped_img, resize)
    if plot is True:
        plt.figure()
        plt.imshow(cropped_img)
        plt.show()
    return cropped_img

def mask_images(mask_fp,img_fp):
    """ returns masked image 
    Args:
        mask_fp (str): filepath to circular mask
        img_fp (str): filepath to SVF image which should have the same dimension as the mask img
    """
    mask = image_to_array(mask_fp)
    img = image_to_array(img_fp)
    assert (img.shape[0] == mask.shape[0]) & (img.shape[1] == mask.shape[1]), "mask and img must have the same w & h dimensions!"
    if len(img.shape) == 3:
        return mask*img
    elif len(img.shape) == 2:
        return mask[:,:,0]*img
    else:
        raise TypeError("Image must have at least 2 dimensions")
    
def pad_images(img,img_dim = 512):
    """ 
    pad images if image column < img_dim
    Args:
        img (np.ndarray): subsetted/cropped image
    """
    nrow,ncol = img.shape[0],img.shape[1]
    if ncol > img_dim:
        raise ValueError(f"Img col is > {img_dim}!")
    if len(img.shape) == 3:
        pad_img = np.zeros((img_dim,img_dim,3))
        pad_img[:,:ncol,:] = img
        pad_img = pad_img.astype(np.uint8)
    else:
        pad_img = np.zeros((img_dim,img_dim))
        pad_img[:,:ncol] = img
        pad_img = pad_img.astype(np.uint8)
        # if np.sum(pad_img) > 255 and np.sum(pad_img)/(img_dim*img_dim*255) < 0.05:
        #     pad_img = np.zeros((img_dim,img_dim)).astype(np.uint8)
    return pad_img#.astype(np.uint8)

def crop_images(img,img_dim = 512):
    """ crop images into 512 x 512 """
    nrow,ncol = img.shape[0],img.shape[1]
    # print(nrow,ncol)
    ncols = ceil(ncol/img_dim)
    nrows = ceil(nrow/img_dim)
    cropped_images = []
    for i in range(nrows):
        for j in range(ncols):
            end_row = (i+1)*img_dim
            if i == nrows -1:
                end_row = nrow
            end_col = (j+1)*img_dim
            if j == ncols -1:
                end_col = ncol
            # print('row: ',i*img_dim,end_row, ', col: ',j*img_dim,end_col)
            # write image info over to padded img
            if len(img.shape) == 3:
                padded_img = np.zeros((img_dim,img_dim,3),dtype=np.uint8)
                padded_img[0:end_row-i*img_dim,0:end_col-j*img_dim,:] = img[i*img_dim:end_row,j*img_dim:end_col,:]
            elif len(img.shape) == 2:
                padded_img = np.zeros((img_dim,img_dim),dtype=np.uint8)
                padded_img[0:end_row-i*img_dim,0:end_col-j*img_dim] = img[i*img_dim:end_row,j*img_dim:end_col]
            else:
                raise TypeError("Image must have at least 2 dimensions")
            cropped_images.append(padded_img)
    return cropped_images

def save_imgs(cut_img_list,filename,save_dir):
    """ save cropped images
    Args:
        cut_img_list (list of np.ndarray): list of cropped images
        filename(str): prefix of the filename to better id the images
        save_dir (str): directory of where to save the cropped images
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i,img in enumerate(cut_img_list):
        save_fp = os.path.join(save_dir,f'{filename}_{str(i).zfill(2)}.png')
        im = Image.fromarray(img)
        im.save(save_fp)
    return

def rotate_image(fp, angle=90, plot = True,save_fp=None):
    """ returns rotated image as np.ndarray
    Args:
        fp (str): filepath of image
        angle (float): angle in deg to rotate the image by
        plot (bool): to plot an original vs rotated im
        save_fp (str): file pathname
    Args:
        np.ndarray: rotated image
    """
    im = Image.open(fp)
    rot_im = im.rotate(angle)
    if plot is True:
        fig, axes = plt.subplots(1,2)
        axes[0].imshow(im)
        axes[0].set_title('Original im')
        axes[1].imshow(rot_im)
        axes[1].set_title(f'Rotated im by {angle}$^o$')
        for ax in axes:
            ax.axis('off')
        plt.show()
    
    if save_fp is not None:
        save_fp = os.path.splitext(save_fp)[0]
        rot_im.save(f'{save_fp}.png')
    return np.asarray(rot_im)

def get_cropped_SVF(mask_fp,img_fp, plot=True, save_dir=None):
    """ 1. mask SVF images with a circular mask, 2. crop SVF images to a square and resize it to 1024 x 1024, 3. then subset each image to 4 tiles, 4. save each tile image
    Args:
        mask_fp (str): filepath of mask
        img_fp (str): filepath of img
        plot (bool): whether to plot the cropped images
        save_dir (str): file directory of where to store the images. filenames and extension are automatically saved
    Returns:
        list of np.ndarray: list of masked, cropped, and resized images
    """
    masked_img = mask_images(mask_fp,img_fp)
    sq_img = crop_square(masked_img, resize=(1024,1024),plot=False)
    cropped_images = crop_images(sq_img,img_dim = 512)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        filename = os.path.splitext(os.path.basename(img_fp))[0]
        save_imgs(cropped_images,filename,save_dir=save_dir)

    if plot is True:
        fig, axes = plt.subplots(2,2,figsize=(25,25))
        for im, ax in zip(cropped_images[:4],axes.flatten()):
            ax.imshow(im)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    return cropped_images

def predicted_to_binary(im,thresh=125):
    """convert predicted rgb image to binary classified image
    Args:
        im (np.ndarray): mxn
    Returns:
        np.ndarray: binary image
    """
    return np.where(im>thresh,1,0).astype(np.uint8)

def stitch_cropped_images(predicted_img_fpList, plot = True):
    """returns predicted stitched_img (np.ndarray): mxn
    Args:
        predicted_img_fpList (list of str) list of filepaths of the predicted img
    Returns:
        np.ndarray: a stitched image from the cropped image
    """
    n_images = len(predicted_img_fpList)
    n_row = n_col = int(n_images**(1/2)) # sqrt number of images to obtain how many rows and cols of cropped images, assuming the original image is a square
    # print(f'ncols {n_col} and nrows {n_row} detected')
    img = image_to_array(predicted_img_fpList[0])
    nrow, ncol = img.shape[0],img.shape[1] # assuming the predicted image is a mxn image
    ndim = (int(nrow*n_row), int(ncol*n_col))
    # print(f'ndim: {ndim}')
    stitched_img = np.zeros(ndim,dtype=np.uint8)
    for i,fp in enumerate(predicted_img_fpList):
        row_index = i//n_row
        col_index = i%n_col
        img = image_to_array(fp)
        # print(f'write to: {row_index*nrow,(row_index+1)*nrow}; {col_index*ncol,(col_index+1)*ncol}')
        stitched_img[row_index*nrow:(row_index+1)*nrow,col_index*ncol:(col_index+1)*ncol] = predicted_to_binary(img[:,:,0])

    if plot is True:
        plt.figure()
        plt.imshow(stitched_img)
        plt.colorbar()
        plt.show()
    return stitched_img


def calculate_SVF(mask_fp,predicted_img, plot = True):
    """ returns SVF in percentage (float)
    Args:
        mask_fp (str): filepath of mask
        img (np.ndarray) predicted image by the trained model, assumed to be 512x512 dimension
        img_fp (str): filepath of img
    Returns:
        float: calculated SVF in percentage
    """
    mask = image_to_array(mask_fp) # binary mask, where pixels==1 corresponds to the entire FOV
    mask = mask[:,:,0]
    sq_mask = crop_square(mask, resize=(1024,1024),plot=False)
    assert sq_mask.shape == predicted_img.shape, f"dimensions of mask {sq_mask.shape} != dimensions of predicted_img {predicted_img.shape}"
    N = np.sum(sq_mask) # FOV pixels
    n = np.sum(predicted_img) # where 1 = sky pixels
    SVF = n/N*100
    if plot is True:
        fig, axes = plt.subplots(1,2,figsize=(10,5))
        axes[0].imshow(sq_mask)
        axes[1].imshow(predicted_img)
        axes[1].set_title(f'SVF: {SVF:.3f} %')
    for ax in axes.flatten():
        ax.axis('off')
    return SVF

def find_latest_checkpoint(checkpoints_path, fail_safe=True):
    """
    Args:
        checkpoints_path (str): name of checkpoint
    Returns:
        str: filepath of latest checkpoint file
    """
    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # print(all_checkpoint_files)
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def model_from_checkpoint_path(checkpoints_path):
    """load model from checkpoint path
    Args:
        checkpoints_path (str): path to checkpoints file
    Returns:
        trained model
    """

    from keras_segmentation.models.all_models import model_from_name
    assert (os.path.isfile(checkpoints_path+"_config.json")
            ), "Checkpoint not found."
    model_config = json.loads(
        open(checkpoints_path+"_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (latest_weights is not None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](
        model_config['n_classes'], input_height=model_config['input_height'],
        input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


