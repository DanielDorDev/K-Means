import numpy
import numpy as np
import init_centroids
import math
from scipy.misc import imread
from multiprocessing import Process
import matplotlib.image as image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def print_cent(cent):  # Function for convert centroid array to printable.
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(
            ' ]', ']').replace(' ', ', ')[1:-1]


class Cluster_Pair:  # Vector, counter pair(for average).
    def __init__(self, vector=0):
        self.counter = 0
        self.num_py = vector

    def __iadd__(self, o):
        self.counter += 1
        self.num_py = self.num_py + o.num_py
        return self

    def average(self):
        return np.divide(self.num_py, self.counter)


class Ex_Means:  # Main class for K-mean algorithm.

    def __init__(self):  # Init array for centroid.
        self.centroid_array = []
        for idx in (2 ** p for p in range(1, 5)):
            self.centroid_array.append(init_centroids.init_centroids(0, idx))

    def train(self, path, process_run):  # Train data using picture.

        picture = imread(path)
        picture = picture.astype(float) / 255.
        img_size = picture.shape
        data = picture.reshape(img_size[0] * img_size[1], img_size[2])

        buffer_output = [str(i) + ".txt" for i in range(4)]
        pool = []
        if process_run:     # Run with process(work faster in computer, little bit faster in u2).
            if __name__ == "__main__":
                for i in range(1, 5)[::-1]:
                    thr = Process(target=self.operate, args=(data, i, buffer_output[i - 1]))
                    pool.append(thr)
                    thr.start()

                for i in range(0, 4)[::-1]:
                    if pool[i].is_alive():
                        pool[i].join()
                    with open(buffer_output[3 - i], 'r') as fin:
                        print(fin.read(), end='')
                exit(0)
        else:       # Run by iterating in the same process.
            for i in range(1, 5):
                self.operate(data, i, buffer_output[i - 1])
             #   with open(buffer_output[i - 1], 'r') as fin:
               #     print(fin.read(), end='')

                picture = imread('dog.jpeg')
                picture = picture.astype(float) / 255.
                img_size = picture.shape

                k_colors = KMeans(n_clusters=2**i).fit(data)
                newimg = k_colors.cluster_centers_[k_colors.labels_]
                newimg = np.reshape(newimg, img_size)
                plt.imsave(str(i) + 'jj.jpeg', newimg)


            exit(0)

    def operate(self, data, idx, stream):  # Calculate and update centroid.
        with open(stream, 'w') as f:  # Print them into file for option of process usage.
            print('k=', 2 ** idx, ':\n', 'iter 0: ', print_cent((self.centroid_array[idx - 1])), sep='', file=f)
            for iterIdx in range(1, 11):
                print('iter ', iterIdx, ': ', print_cent(self.update(data, 2 ** idx)), sep='', file=f)

    def update(self, data, k_number):  # Update the clusters, calculate distance and cluster to array.
        """

        :rtype: object - ndarray
        """
        array_index = int(math.log2(k_number)) - 1  # Calculate array index for centroid(from given k).

        if type(data) == numpy.ndarray and type(k_number) == int:

            cluster = [Cluster_Pair() for x in range(k_number)]  # Init default clusters(with 0).

            for from_data in data:  # Calculate the distance using lianlg.norm.
                distances = [np.linalg.norm(from_data - cent) for cent in self.centroid_array[array_index]]
                cluster[distances.index(min(distances))] += (Cluster_Pair(from_data))

            for idx, centroid in enumerate(self.centroid_array[array_index]):
                if cluster[idx].counter:  # Update points if cluster exist.
                    self.centroid_array[array_index][idx] = cluster[idx].num_py / cluster[idx].counter
        return self.centroid_array[array_index]


kMean = Ex_Means()
kMean.train('dog.jpeg', False)      # True - run by processes, False - run by iterate.
