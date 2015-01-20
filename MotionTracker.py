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

#Define a function to build a list of stable feature candidates:.
def get_stable_feature_candidates(frame,patch_radius):
    features = corner_peaks(corner_harris(frame), patch_radius)
    return features

#Define a function to delete overlapping patches:
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
    #Find gradients:
    gradx = hsobel(frame)
    grady = vsobel(frame)

    #Calculate the w:
    w = np.array([[None]*(patch_radius*2+1)]*(patch_radius*2+1))
    for i in range(-patch_radius, patch_radius+1):
        for j in range(-patch_radius, patch_radius+1):
            w[i+patch_radius,j+patch_radius] = w_func(i,j)

    #Precompute values through numpy:
    Gxx = np.multiply(gradx, gradx)
    Gxy = np.multiply(gradx, grady)
    Gyy = np.multiply(grady, grady)

    #Calculate the G matrix of every feature:
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

#Define a function to calculate residue-scaled gradients:
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

#Define a function to calculate a single residue-scaled gradient:
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
def build_feature_sequence(video_sequence, patch_radius=2, eigenvalue_threshold=0.001, w_func=lambda x,y:1, update_rate = 10):
    #Define the list of locations:
    patch_locations = [None]*len(video_sequence)

    #Find some features to track:
    patch_locations[0] = get_stable_feature_candidates(video_sequence[0],patch_radius)
    patch_locations[0] = remove_overlap(patch_locations[0], patch_radius)
    xmax = len(video_sequence[0])
    ymax = len(video_sequence[0][0])

    for i in range(1,len(video_sequence)):
        #Calculate G and e:
        G = calculate_structure_tensors(video_sequence[i], patch_locations[i-1], patch_radius, w_func)
        e = calculate_residue_scaled_gradients(patch_locations[i-1],video_sequence[i-1], video_sequence[i], patch_radius, w_func)

        deletion_indexes = []
        for j in range(len(G)):
            eigenvalues = np.linalg.eig(G[j])[0]
            if min(eigenvalues) < 0.0002:
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
            patch_locations[i] = []

        #Add in new patch locations:
        if i%update_rate == 0:
            new_feats = get_stable_feature_candidates(video_sequence[i],patch_radius)
            if patch_locations[i] == []:
                patch_locations[i] = new_feats
            else:
                patch_locations[i] = np.concatenate((patch_locations[i], new_feats), axis=0)
            patch_locations[i] = remove_overlap(patch_locations[i], patch_radius)

    return patch_locations


'''
Illustration functions:
'''

#Define a function to paint a list of patches into a video sequence:
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
Interface
'''

def track(image_sequence_path, image_limit=400, w_func='uniform', patch_radius=15, eigenvalue_threshold=0.001, update_rate = 20):

    #Do some casting because python fails:
    patch_radius = int(patch_radius)
    image_limit = int(image_limit)
    eigenvalue_threshold = float(eigenvalue_threshold)
    update_rate = int(update_rate)

    if w_func == 'gaussian':
        w_func = lambda x,y: 1/(2.0*math.pi*(1.41**2))*math.exp(-((x-4-1)**2 + (y-4-1)**2)/(2.0*(1.41**2)))
    elif w_func=='uniform':
        w_func = lambda x,y: 1

    #Read in the images:
    images = VideoLoader.load_images_at_path(image_sequence_path, limit=image_limit)

    #Scale to a 0-1 representation:
    images = [np.multiply(i, 1/255.0) for i in images]

    #Build a sequence of feature locations:
    feat_seq = build_feature_sequence(images, patch_radius=patch_radius, eigenvalue_threshold=eigenvalue_threshold, w_func=w_func, update_rate = update_rate)

    #Paint the features on top of the images and return the result:
    painted = [paint_patches(images[x],feat_seq[x], patch_radius, [1,1,0]) for x in range(image_limit)]
    return painted

'''
Testing playground:
'''

import math
if __name__ == '__main__':
    fn1 = input("Write the name of the folder containing the image sequence:\n")

    query = input("Specify any additional parameters, or write \"done\" to proceed. Default:\nimage_limit=400, w_func='uniform', patch_radius=15, eigenvalue_threshold=0.001, update_rate = 20\n")

    if query == "done":
        p=track(fn1)
    else:
        ql = query.strip().split(" ")

        ql2 = [s.split("=") for s in ql]
        qld = {s[0]: s[1] for s in ql2}

        p=track(fn1, **qld)

    a = input("Do you wish to save the result? [Y/N]")

    if a == 'Y':
        save = True
    else:
        save = False

    VideoLoader.animate(p, save=save)
