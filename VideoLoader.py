__author__ = 'Michael'

'''
Imports:
'''
import os
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
'''
Utility functions:
'''
#Define a function to get all paths to pgm files in a folder:
def get_pgm_paths(a_dir):
    return [a_dir+'/'+name for name in os.listdir(a_dir) if (os.path.isfile(os.path.join(a_dir, name)) and name.endswith('.pgm'))]

'''
Interface:
'''

#Define a function to load a sequence of images from a folder:
def load_images_at_path(path, limit=None):
    #Gather the paths:
    paths = get_pgm_paths(path)

    #If requested, limit the number of images:
    if limit is not None:
        paths = paths[:limit]

    #Read in the images and return the result:
    items = [rgb2gray(imread(p)) for p in paths]
    return items


#TODO: Maybe avoid memory errors for limit > 400
#Define a function to show a sequence of images as an animation
def animate(image_sequence):
    fig = plt.figure()

    ims = []
    for i in xrange(len(image_sequence)):
        im = plt.imshow(image_sequence[i],cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255)
        ims.append([im])

    ani=animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)

    plt.show()

'''
Testing playground:
'''

if __name__ == '__main__':

    ims = load_images_at_path('DudekSeq',limit=400)

    print "Done loading"

    animate(ims)
