__author__ = 'Michael'

'''
Imports:
'''

import VideoLoader
import numpy as np
from skimage.filter import hsobel,vsobel
from skimage.feature import corner_harris, corner_peaks
from skimage.color import gray2rgb
from skimage.viewer import ImageViewer
import itertools
'''
Feature functions:
'''
#Builds a list of stable feature candidates. These are basically Harris corners.
def get_stable_feature_candidates(frame,patch_radius, eigenvalue_threshold, w_func=lambda x,y: 1):
    #TODO: Use library implementation of harris corner detection instead of this
    '''gradx = hsobel(frame)
    grady = vsobel(frame)

    xmax = len(frame)
    ymax = len(frame[0])

    features = []

    w = np.array([[None]*(patch_radius*2+1)]*(patch_radius*2+1))

    #Calculate the w:
    for i in range(-patch_radius, patch_radius+1):
        for j in range(-patch_radius, patch_radius+1):
            w[i+patch_radius,j+patch_radius] = w_func(i,j)

    Gxx = np.multiply(gradx, gradx)
    Gxy = np.multiply(gradx, grady)
    Gyy = np.multiply(grady, grady)

    for x in xrange(xmax):
        for y in xrange(ymax):
            if patch_radius-1 < x < xmax-patch_radius and patch_radius-1 < y < ymax-patch_radius:
                G = calculate_structure_tensor(Gxx, Gxy, Gyy, x,y, patch_radius,w)
                eigenvalues = np.linalg.eig(G)[0]

                if min(eigenvalues) > eigenvalue_threshold:
                    features.append((x,y))
    '''
    features = corner_peaks(corner_harris(frame), patch_radius)
    features = remove_overlap(features, patch_radius)
    return features

def remove_overlap(coords, patch_radius):
    l =[]
    for i,j in itertools.combinations(range(len(coords)), 2):
        if coords[i][0]-(2*patch_radius) < coords[j][0] < coords[i][0]+(2*patch_radius) and coords[i][1]-(2*patch_radius) < coords[j][1] < coords[i][1]+(2*patch_radius):
            l.append(j)
    coords = np.delete(coords, l, axis=0)

    return coords

'''
Mathematical functions:
'''
#Define a function to create structure tensors for a list of points:
def calculate_structure_tensors(frame, feature_list, patch_radius, w_func=lambda x,y:1):
    gradx = hsobel(frame)
    grady = vsobel(frame)

    w = np.array([[None]*(patch_radius*2+1)]*(patch_radius*2+1))

    #Calculate the w:
    for i in range(-patch_radius, patch_radius+1):
        for j in range(-patch_radius, patch_radius+1):
            w[i+patch_radius,j+patch_radius] = w_func(i,j)

    Gxx = np.multiply(gradx, gradx)
    Gxy = np.multiply(gradx, grady)
    Gyy = np.multiply(grady, grady)

    G_list = np.array([None]*len(feature_list))

    for i,(x,y) in enumerate(feature_list):
        G_list[i] = calculate_structure_tensor(Gxx, Gxy, Gyy, x, y, patch_radius, w)

    return G_list

#Define a function to calculate a single structure tensor:
def calculate_structure_tensor(Gxx, Gxy, Gyy, x, y, patch_radius, w):
    G00 = 0
    G01 = 0
    G11 = 0
    for u in xrange(-patch_radius, patch_radius):
        for v in xrange(-patch_radius, patch_radius):
            xs = x+u
            ys = y+v

            G00 += Gxx[xs,ys]*w[u,v]
            G01 += Gxy[xs,ys]*w[u,v]
            G11 += Gyy[xs,ys]*w[u,v]

    return np.array([[G00,G01],[G01,G11]])


def calculate_residue_scaled_gradients(patches, old_frame, new_frame, patch_radius, w_func=lambda x,y:1):
    #Calculate some gradients:
    gradx = hsobel(new_frame)
    grady = vsobel(new_frame)

    #Calculate the scaled gradients:
    diff = np.subtract(old_frame, new_frame)
    ex = np.multiply(diff, gradx)
    ey = np.multiply(diff, grady)

    w = np.array([[None]*(patch_radius*2+1)]*(patch_radius*2+1))

    #Calculate the w:
    for i in range(-patch_radius, patch_radius+1):
        for j in range(-patch_radius, patch_radius+1):
            w[i+patch_radius,j+patch_radius] = w_func(i,j)

    #Get the scaled gradient at each point:
    e_list = np.array([None]*len(patches))

    for i,(x,y) in enumerate(patches):
        e_list[i] = calculate_residue_gradient(ex, ey, x, y, patch_radius, w)

    return e_list

def calculate_residue_gradient(ex, ey, x, y, patch_radius, w):
    e1 = 0
    e2 = 0
    for u in range(-patch_radius, patch_radius+1):
        for v in range(-patch_radius, patch_radius+1):
            xs = x+u
            ys = y+v

            e1 += ex[xs, ys]*w[u,v]
            e2 += ey[xs, ys]*w[u,v]

    return np.array([e1,e2])


'''
Motion tracking functions:
'''

#Define a function to build a feature sequence matching a video sequence:
def build_feature_sequence(video_sequence, patch_radius=2, eigenvalue_threshold=0.3, w_func=lambda x,y:1, update_rate = 10):
    #Define the list of locations:
    patch_locations = [None]*len(video_sequence)

    #Find some features to track:
    patch_locations[0] = get_stable_feature_candidates(video_sequence[0],patch_radius,eigenvalue_threshold,w_func)
    xmax = len(video_sequence[0])
    ymax = len(video_sequence[0][0])

    view = ImageViewer(paint_patches(video_sequence[0],patch_locations[0],patch_radius, [1,1,0]))
    view.show()
    for i in range(1,len(video_sequence)):
        if i%10 == 0:
            print i

        #Calculate G and e:
        G = calculate_structure_tensors(video_sequence[i], patch_locations[i-1], patch_radius, w_func)
        e = calculate_residue_scaled_gradients(patch_locations[i-1],video_sequence[i-1], video_sequence[i], patch_radius, w_func)

        deletion_indexes = []
        for j in range(len(G)):
            eigenvalues = np.linalg.eig(G[j])[0]
            if min(eigenvalues) < 0.001:
                deletion_indexes.append(j)

        G = np.delete(G, deletion_indexes, axis=0)
        e = np.delete(e, deletion_indexes, axis=0)
        feature_view = patch_locations[i-1]
        feature_view = np.delete(feature_view, deletion_indexes, axis=0)

        #Update the patches:
        displacements = [np.dot(np.linalg.inv(G[j]),e[j]) for j in range(len(G))]

        if len(G) > 0:
            patch_locations[i] = np.add(feature_view, displacements)
            patch_locations[i] = [(x,y) for (x,y) in patch_locations[i] if patch_radius <= x < xmax-patch_radius and patch_radius <= y < ymax-patch_radius]
        else:
            patch_locations[i] = np.array([])

        #Add in new patch locations:
        if i%update_rate == 0:
            new_feats = get_stable_feature_candidates(video_sequence[i],patch_radius,eigenvalue_threshold,w_func)
            patch_locations[i] = np.concatenate((patch_locations[i], new_feats), axis=0)
            patch_locations[i] = remove_overlap(patch_locations[i], patch_radius)

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
import itertools
import math
if __name__ == '__main__':
    ims = VideoLoader.load_images_at_path("DudekSeq", limit=400)
    ims = [np.multiply(i, 1/255.0) for i in ims]

    feat_seq = build_feature_sequence(ims,patch_radius=4, w_func=lambda x,y: 1/(2.0*math.pi*(1.41**2))*math.exp(-((x-4-1)**2 + (y-4-1)**2)/(2.0*(1.41**2))))

    painted = [paint_patches(ims[x],feat_seq[x], 4, [1,1,0]) for x in range(400)]

    VideoLoader.animate(painted)
