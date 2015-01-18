__author__ = 'Michael'

'''
Imports:
'''

import VideoLoader
import numpy as np
from skimage.filter import hsobel,vsobel
from skimage.feature import corner_harris
from skimage.color import gray2rgb
from skimage.viewer import ImageViewer
'''
Feature functions:
'''
#Builds a list of stable feature candidates. These are basically Harris corners.
def get_stable_feature_candidates(frame,patch_radius, eigenvalue_threshold, w_func=lambda x,y: 1):
    #TODO: Use library implementation of harris corner detection instead of this
    gradx = hsobel(frame)
    grady = vsobel(frame)

    xmax = len(frame)
    ymax = len(frame[0])

    features = []

    ws = np.array([[w_func(x,y) for y in xrange(ymax)] for x in xrange(xmax)])
    Gxx = np.multiply(np.multiply(gradx, gradx),ws)
    Gxy = np.multiply(np.multiply(gradx, grady),ws)
    Gyy = np.multiply(np.multiply(grady, grady),ws)

    for x in xrange(xmax):
        print x
        for y in xrange(ymax):
            if patch_radius-1 < x < xmax-patch_radius and patch_radius-1 < y < ymax-patch_radius:
                G = calculate_structure_tensor(Gxx, Gxy, Gyy, x,y, patch_radius)
                eigenvalues = np.linalg.eig(G)[0]

                if min(eigenvalues) > eigenvalue_threshold:
                    features.append((x,y))

    return features

'''
Mathematical functions:
'''
#Define a function to create structure tensors for a list of points:
def calculate_structure_tensors(frame, feature_list, patch_radius, w_func=lambda x,y:1):
    gradx = hsobel(frame)
    grady = vsobel(frame)

    xmax = len(gradx)
    ymax = len(gradx[0])

    ws = np.array([[w_func(x,y) for y in xrange(ymax)] for x in xrange(xmax)])
    Gxx = np.multiply(np.multiply(gradx, gradx),ws)
    Gxy = np.multiply(np.multiply(gradx, grady),ws)
    Gyy = np.multiply(np.multiply(grady, grady),ws)

    G_list = []

    for x,y in feature_list:
        calculate_structure_tensor(Gxx, Gxy, Gyy, x, y, patch_radius, w_func)

    return G_list

#Define a function to calculate a single structure tensor:
def calculate_structure_tensor(Gxx, Gxy, Gyy, x, y, patch_radius):
    G00 = 0
    G01 = 0
    G11 = 0
    for u in xrange(-patch_radius, patch_radius):
        for v in xrange(-patch_radius, patch_radius):
            xs = x+u
            ys = y+v

            G00 += Gxx[xs,ys]
            G01 += Gxy[xs,ys]
            G11 += Gyy[xs,ys]

    return np.array([[G00,G01],[G01,G11]])


def calculate_residue_scaled_gradients(patches, old_frame, new_frame, patch_radius, w_func):
    pass

'''
Motion tracking functions:
'''

#Define a function to build a feature sequence matching a video sequence:
def build_feature_sequence(video_sequence, patch_radius=2, w_func=lambda x,y:1):
    #Define the list of locations:
    patch_locations = [None]*len(video_sequence)

    #Find some features to track:
    patch_locations[0] = get_stable_feature_candidates(video_sequence[0],patch_radius)


    for i in xrange(1,len(video_sequence)):
        #TODO: Maybe we want to update with new non-overlapping candidates (by rerunning get_features and checking overlap) every 15 or so iterations (expensive)

        #Calculate G and e:
        G = calculate_structure_tensors(video_sequence[i], patch_locations[i-1], patch_radius, w_func)
        e = calculate_residue_scaled_gradients(patch_locations[i-1],video_sequence[i-1], video_sequence[i], patch_radius, w_func)

        #TODO: Discard numerically unstable patches

        #Update the patches:
        displacements = [np.dot(np.linalg.inv(G[i]),e[i]) for i in range(len(G))]
        patch_locations[i] = np.add(patch_locations[i-1], displacements)

    return patch_locations

'''
Illustration functions:
'''

def paint_patches(frame, patches, patch_radius, color):
    col_frame = gray2rgb(frame)
    for i in xrange(len(patches)):
        x,y = patches[i]

        col_frame[x-patch_radius:x+patch_radius, y-patch_radius:y-patch_radius+1] = color
        col_frame[x-patch_radius:x+patch_radius, y+patch_radius:y+patch_radius+1] = color
        col_frame[x-patch_radius:x-patch_radius+1, y-patch_radius:y+patch_radius] = color
        col_frame[x+patch_radius:x+patch_radius+1, y-patch_radius:y+patch_radius] = color

    return col_frame


'''
Testing playground:
'''

if __name__ == '__main__':
    img = VideoLoader.load_images_at_path('DudekSeq',limit=1)[0]
    feats = get_stable_feature_candidates(img, 4, 0.1)
    p = paint_patches(img, feats, 4, [255,255,0])
    view = ImageViewer(img)
    view.show()
    view = ImageViewer(p)
    view.show()


