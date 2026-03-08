import matplotlib.pyplot as plt
import cv2  #image processing
import numpy as np
from skimage.morphology import skeletonize  #brings down the thickness
import networkx as nx  #for analyzing nodes

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #image loaded into 2D grayscale Numpy array(0,.255)
    if img is None:
        raise ValueError("Failed to load image")
    ul, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  #converts grayscale image into an inverted binary image
    return binary  #array with values 0 or 255

def skeletonize_image(binary):
    skel = skeletonize(binary // 255).astype(np.uint8)  #first 0/255 converted to 0/1 because skeletonize, then converted to uint8
    return skel

def graph(skel):
    #list of relative coordinates for neighbourhood
    close = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)]
    h, w = skel.shape    #shape of skeleton
    g = nx.Graph()     #empty undirected graph

    #nodes being added over all of x,y
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 1:
                g.add_node((x, y))

    #check 8-neighbourhood and add edges
    for y in range(h):
        for x in range(w):
            if skel[y, x] == 1:
                for dx, dy in close:
                    if 0 <= x + dx < w and 0 <= y + dy < h:
                        if skel[y + dy, x + dx] == 1:
                            g.add_edge((x, y), (x + dx, y + dy))
    #plt.plot(g), plt.show()
    return g

def find_crossings(g,r):
    crossings = []
    for node in g.nodes():
        deg = g.degree(node)      #number of neighbors
        if deg >= 4:
            crossings.append(node)

    # remove near-duplicate crossings within radius
    filtered = []
    for c in crossings:
        if all(np.hypot(c[0] - fc[0], c[1] - fc[1]) > r for fc in filtered):   #distance from node checking
            filtered.append(c)
    #plt.plot(filtered), plt.show()
    return filtered

def main(image,r):
    binary = preprocess_image(image)
    skel = skeletonize_image(binary)
    g = graph(skel)
    crossings = find_crossings(g,r)
    return len(crossings)


image_path = "C:\\Users\\kittu\\OneDrive\\Desktop\\Project\\Code\\image.png"
print("Hello, I will find your crossing number!!!!!!!!!")
r = 5
while True:
    result = main(image_path,r)
    print("Is", result, "your crossing number?...yes/no")
    if input().capitalize() == "Yes":
        print("YAYYY!!!!!!!!")
        break
    else:
        if r == 0:
            break
        else:
            print("Hmm... I see. Let me try again....")
            r -= 1





