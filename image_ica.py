from sklearn.decomposition import FastICA
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import operator
import os

image_folder = 'images'
output_folder = 'images/res'
melo_pics = ['meloetta-aria.jpg', 'meloetta-pirouette.jpg']

shift = (30, 0)
alphas = [0.25, 0.75]


def mix_images(img1, img2, alpha):
    nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)
    newimg1 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg1.paste(img2, shift)
    newimg1.paste(img1, (0, 0))

    # paste img2 on top of img1
    newimg2 = Image.new('RGBA', size=(nw, nh), color=(0, 0, 0, 0))
    newimg2.paste(img1, (0, 0))
    newimg2.paste(img2, shift)

    # blend with alpha=0.5
    result = Image.blend(newimg1, newimg2, alpha=alpha)
    return result


def fastica(mixed_img, width, height, n):
    flat_imgs = []
    restored_imgs = []

    for img in mixed_img:
        img = np.array(img)
        flat_imgs.append(img.flatten())
        
    # test = list(zip(flat_imgs[0], flat_imgs[1]))
    test = list(zip(*flat_imgs))

    fast_ica  = FastICA(n_components=n)
    separated = fast_ica.fit_transform(test)

    for i in range(separated.shape[1]):
        restored_imgs.append(np.reshape(separated[:,i], (height, width)))

    fig, axes = plt.subplots(2, len(alphas), figsize=(8,8))
    for index, img in enumerate(restored_imgs):
        img_neg = np.ones([height, width]) - img
        axes[0][index].imshow(img, cmap='gray')
        axes[0][index].set_title('Image {}'.format(index+1))
        axes[1][index].imshow(img_neg, cmap='gray')
        axes[1][index].set_title('Image {} (neg)'.format(index+1))

    fig.suptitle('ICA Results ({} Images)'.format(len(alphas)))
    fig.savefig(os.path.join(output_folder, 'ica_res_{}.png'.format(len(alphas))), dpi=500, format='png')

    plt.show()


if __name__ == "__main__":
    # main()
    image_1 = Image.open(os.path.join(image_folder, melo_pics[0]))
    image_2 = Image.open(os.path.join(image_folder, melo_pics[1]))

    mixed_imgs = []
    # mixed_imgs.append(mix_images(image_1, image_2, 0.25).convert('L'))
    # mixed_imgs.append(mix_images(image_1, image_2, 0.75).convert('L'))

    for alpha in alphas:
        mixed_imgs.append(mix_images(image_1, image_2, alpha).convert('L'))

    mixed_imgs[0].show()
    mixed_imgs[1].show()
    width, height = mixed_imgs[0].size
    n = len(mixed_imgs)

    fastica(mixed_imgs, width, height, n)