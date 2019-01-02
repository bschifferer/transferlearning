import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def add_border(img, image_shape):
    border_v = 0
    border_h = 0
    if (image_shape[1]/image_shape[0]) >= (img.shape[0]/img.shape[1]):
        border_v = int((((image_shape[1]/image_shape[0])*img.shape[1])-img.shape[0])/2)
    else:
        border_h = int((((image_shape[0]/image_shape[1])*img.shape[0])-img.shape[1])/2)
    img = cv2.copyMakeBorder(img, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
    return(img)
        
def open_image(FILE, image_shape, padding = True):
    img = cv2.imread(FILE, cv2.IMREAD_COLOR)
    if img is None:
        return(np.zeros((image_shape[0], image_shape[1], image_shape[2]), np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if padding:
        img = add_border(img, image_shape)
    img = cv2.resize(img, (image_shape[0], image_shape[1]))
    return(np.asarray(img))

def load_images(PATH, list_images, image_shape, padding = True):
    images = [open_image(PATH + image, image_shape, padding) for image in list_images]
    return(np.asarray(images))

def list_all_images(PATH, SUB = ''):
    elements = os.listdir(PATH + SUB)
    images_subfolders = [list_all_images(PATH, element) for element in elements if os.path.isdir(PATH + element)]
    images_subfolders = [element for subfolder in images_subfolders for element in subfolder ]
    images = [SUB + '/' + element for element in elements if element.endswith('.png') or element.endswith('.jpg')]
    return(images + images_subfolders)

def visualize_reco(reco, index_itemset, image_names, path):
    N = len(reco)
    K = reco[0].shape[0]
    f, axarr = plt.subplots(N, K+1)
    f.set_figheight(2*N)
    f.set_figwidth(3*K)
    for n in range(0, N):
        if n == 0:
            axarr[n, 0].set_title("Basis")
        axarr[n, 0].imshow(open_image(path + image_names[index_itemset[n]], [299, 299, 3]))
        axarr[n, 0].set_yticklabels([])
        axarr[n, 0].set_xticklabels([])
        for k in range(0, K):
            if n == 0:
                axarr[n, k+1].set_title("Reco" + str(k+1))
            axarr[n, k+1].imshow(open_image(path + image_names[reco[n][k]], [299, 299, 3]))
            axarr[n, k+1].set_yticklabels([])
            axarr[n, k+1].set_xticklabels([])