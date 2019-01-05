import numpy as np
from umap import UMAP
from sklearn.cluster import KMeans
from keras.applications import vgg16
import skimage

def random_strategy(data, initial_data_size, nb_of_classes):
    data_size = data.shape[0]
    return np.random.randint(0, data_size, initial_data_size)
    
    
def the_furthest_in_cluster(data, initial_data_size, nb_of_classes):
    data_resized = np.array([skimage.transform.resize(image, (48, 48, 3)) for image in data])
    feature_model = vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (48, 48, 3))
    featured_data = feature_model.predict(data_resized/.255).squeeze()
    
    clust_model = KMeans(n_clusters=nb_of_classes)
    clust_model.fit(featured_data)
    predicted_labels = clust_model.predict(featured_data)
    predicted_mapping = list(zip(featured_data, predicted_labels))
    
    cutoff = initial_data_size // nb_of_classes
    idx_out = []
    
    for i in range(nb_of_classes):
        centroid = clust_model.cluster_centers_[i]
        data_in_cluster = [(a, point) for a, (point, label) in enumerate(predicted_mapping) if label == i]
        idx, data = zip(*data_in_cluster)
        dist = np.linalg.norm(centroid - np.array(data), axis=1)
        sorted_data = sorted(zip(idx, dist), key=lambda x:x[1], reverse=True)
        sorted_data = list(sorted_data)[:cutoff]
        idx, _ = zip(*sorted_data)
        idx_out += idx
        
    idx_out = np.array(idx_out)
    np.random.shuffle(idx_out)
    return idx_out


def the_closest_in_cluster(data, initial_data_size, nb_of_classes):
    data_resized = np.array([skimage.transform.resize(image, (48, 48, 3)) for image in data])
    feature_model = vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (48, 48, 3))
    featured_data = feature_model.predict(data_resized/.255).squeeze()
    
    clust_model = KMeans(n_clusters=nb_of_classes)
    clust_model.fit(featured_data)
    predicted_labels = clust_model.predict(featured_data)
    predicted_mapping = list(zip(featured_data, predicted_labels))
    
    cutoff = initial_data_size // nb_of_classes
    idx_out = []
    
    for i in range(nb_of_classes):
        centroid = clust_model.cluster_centers_[i]
        data_in_cluster = [(a, point) for a, (point, label) in enumerate(predicted_mapping) if label == i]
        idx, data = zip(*data_in_cluster)
        dist = np.linalg.norm(centroid - np.array(data), axis=1)
        sorted_data = sorted(zip(idx, dist), key=lambda x:x[1])
        sorted_data = list(sorted_data)[:cutoff]
        idx, _ = zip(*sorted_data)
        idx_out += idx
        
    idx_out = np.array(idx_out)
    np.random.shuffle(idx_out)
    return idx_out


def middle_in_cluster(data, initial_data_size, nb_of_classes):
    data_resized = np.array([skimage.transform.resize(image, (48, 48, 3)) for image in data])
    feature_model = vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (48, 48, 3))
    featured_data = feature_model.predict(data_resized/.255).squeeze()
    
    clust_model = KMeans(n_clusters=nb_of_classes)
    clust_model.fit(featured_data)
    predicted_labels = clust_model.predict(featured_data)
    predicted_mapping = list(zip(featured_data, predicted_labels))
    
    cutoff = initial_data_size // nb_of_classes
    cutoff = cutoff // 2
    idx_out = []
    
    for i in range(nb_of_classes):
        centroid = clust_model.cluster_centers_[i]
        data_in_cluster = [(a, point) for a, (point, label) in enumerate(predicted_mapping) if label == i]
        idx, data = zip(*data_in_cluster)
        dist = np.linalg.norm(centroid - np.array(data), axis=1)
        sorted_data = list(sorted(zip(idx, dist), key=lambda x:x[1]))
        middle = len(sorted_data) // 2
        
        sorted_data1 = sorted_data[middle - cutoff: middle]
        sorted_data2 = sorted_data[middle: middle + cutoff]
        idx1, _ = zip(*sorted_data1)
        idx2, _ = zip(*sorted_data2)
        idx_out += idx1
        idx_out += idx2
        
    idx_out = np.array(idx_out)
    np.random.shuffle(idx_out)
    return idx_out




#def the_furthest_in_cluster(data, init_data_size, nb_of_classes):
# 	# problem z danymi nD gdzie n>1.
# 	# w ostatecznosci zaimplementuj klastrowanie wlasnorecznie z norma macierzowa, tylko jaka?
# 	# obecnie wyciaganie cech z obrazow za pomoca przetrzenowanego vgg16
# 	new_shape = (48,48,3)
# 	x_train_resized = np.asarray([transform.resize(image, new_shape) for image in data])
# 	vgg16_model = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(48,48,3))
# 	featured_data = vgg16_model.predict(x_train_resized/.255)
# 	featured_data  = featured_data.reshape(data.shape[0], -1)
# 	##OBLICZ CENTROIDY
# 	clust_algo = KMeans(n_clusters=nb_of_classes)
# 	clust_algo.fit(predicted_data)
# 	predicted_labels = clusters.predict(featured_data)
# 	pairs = zip(featured_data, predicted_labels)

# 	cutoff = init_data_size // nb_of_classes
# 	gathered_data = []

# 	## WEZ PUNKTY NAJDALSZE OD NIEGO W TYM KLASTRZE
# 	for i in range(nb_of_classes):
# 		centroid = clusters.cluster_centers_[i]
# 		class_data = [(i,pict) for i, pict, label in enumerate(pairs) if label==i]
# 		distance_to_centroid = [(el[0], np.linalg.norm(centroid, el[1])) for el in class_data]
# 		more_informative_data = sorted(distance_to_centroid, key=lambda x: x[1], reversed=True)
# 		idxs ,_ = zip(*more_informative_data[:cutoff])  
# 		gathered_data += data[idxs]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

# 	return gathered_data

# 	#NIE MOGE TEGO PRZESKOCZYC, SKUP SIE
