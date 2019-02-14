import h5py
import PIL
import scipy
import scipy.io as sio
import scipy.ndimage.interpolation
import scipy.misc
import numpy as np
from random import uniform

nyu_depth = h5py.File('./nyu_depth_v2_labeled.mat', 'r')

image = nyu_depth['images']
print(len(image))
depth = nyu_depth['depths']

# img = image[1,:,:,:].astype(float)
# img = np.swapaxes(img, 0, 2)
# img = scipy.misc.imresize(img, [480, 640]).astype(float)
# scipy.misc.imsave('./haze.jpg', img)

print(depth[0].shape)


def get_image(one):
    img = one.astype(float)
    img = np.swapaxes(img, 0, 2)
    return img

def get_depth(one):
    maxhazy = one.max()
    minhazy = one.min()
    print(maxhazy, minhazy)
    img = (one) / (maxhazy)

    img = np.swapaxes(img, 0, 1)

    return img

def main():
    index = 10
    img = get_image(image[index])
    scipy.misc.imsave('./img.bmp', img)

    depth_img = get_depth(depth[index])
    scipy.misc.imsave('./depth.jpg', depth_img)

    # scale1 = (depth_img.shape[0]) / 480
    # scale2 = (depth_img.shape[1]) / 640

    # print(scale1, scale2)

    # gt_depth = scipy.ndimage.zoom(depth_img, (1 / scale1, 1 / scale2), order=1)
    # scipy.misc.imsave('./depth2.jpg', gt_depth)

    beta = uniform(0.5, 2)

    tx1 = np.exp(-beta * depth_img)

    a = 1 - 0.5 * uniform(0, 1)

    print('beta', beta)
    print('a', a)
    # A = [a,a,a]


    #beta
    bias = 0.05
    temp_beta = 0.4 + 0.2*1
    beta = uniform(temp_beta-bias, temp_beta+bias)
    print('bera', beta)

    tx1 = np.exp(-beta * depth_img)
            
    #A
    abias = 0.1
    temp_a = 0.5 + 0.2*1
    a = uniform(temp_a-abias, temp_a+abias)
    print('a', a)
    A = [a,a,a]

    m = img.shape[0]
    n = img.shape[1]

    rep_atmosphere = np.tile(np.reshape(A, [1, 1, 3]), [m, n, 1])
    tx1 = np.reshape(tx1, [m, n, 1])

    max_transmission = np.tile(tx1, [1, 1, 3])

    haze_image = img * max_transmission + rep_atmosphere * (1 - max_transmission)

    scipy.misc.imsave('./haze.jpg', haze_image)

if __name__ == "__main__":
    main()
    pass
