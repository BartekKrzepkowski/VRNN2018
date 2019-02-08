import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

def picker_simple(model, data, labeled_idx, cutoff):
    start = len(labeled_idx)
    end = start + cutoff
    return np.concatenate([labeled_idx, np.arange(start, end)])

##zmien jutrp
def picker_max_min(model, data, labeled_idx, cutoff):
    unlabeled_idx = np.setdiff1d(np.arange(data.shape[0]), labeled_idx)
    predicted_data = model.predict(data[unlabeled_idx])
    try:
        pairs = enumerate(np.max(predicted_data, axis=1))
    except:
        print(predicted_data)
    idx, _ = zip(*sorted(pairs, key=lambda x:x[1])[:cutoff])
    idx = unlabeled_idx[list(idx)]
    labeled_idx = np.concatenate([labeled_idx, idx])
    return labeled_idx

def picker_margin(model, data, labeled_idx, cutoff):
    unlabeled_idx = np.setdiff1d(np.arange(data.shape[0]), labeled_idx)
    predicted_data = model.predict(data[unlabeled_idx])
    max_in_dist = np.max(predicted_data, axis=1)
    min_in_dist = np.min(predicted_data, axis=1)
    pairs = enumerate(max_in_dist-min_in_dist)
    idx, _ = zip(*sorted(pairs, key=lambda x:x[1])[:cutoff])
    idx = unlabeled_idx[list(idx)]
    labeled_idx = np.concatenate([labeled_idx, idx])
    return labeled_idx
    
def entropy_10(tab):
    return entropy(tab, base=10)

def picker_entropy(model, data, labeled_idx, cutoff):
    unlabeled_idx = np.setdiff1d(np.arange(data.shape[0]), labeled_idx)
    predicted_data = model.predict(data[unlabeled_idx])
    entropies = list(map(entropy_10, predicted_data))
    pairs = enumerate(entropies)
    idx, _ = zip(*sorted(pairs, key=lambda x: x[1], reverse=True)[:cutoff])
    idx = unlabeled_idx[list(idx)]
    labeled_idx = np.concatenate([labeled_idx, idx])
    return labeled_idx

def sampling(data_idx, n):
    array_of_samples = []
    size_of_sampling = len(data_idx) // 2
    for _ in range(n):
        samples = list(set(np.random.choice(data_idx, size_of_sampling, replace=True)))
        samples = np.array(samples)
        array_of_samples.append(samples)
    return array_of_samples

def picker_ensemble(build_model, data, labeled_idx, cutoff):
    predicted_datas = []
    samples = sampling(labeled_idx, 5)
    unlabeled_idx = np.setdiff1d(np.arange(data[0].shape[0]), labeled_idx)
    
    for data_idx in tqdm(samples):
        model = build_model()
        model.fit(x=data[0][data_idx],
                  y=data[1][data_idx],
                  epochs=5,
                  batch_size=128
                 )
        predicted_data = model.predict(data[0][unlabeled_idx])
        predicted_datas.append(np.argmax(predicted_data, axis=1))
        
    unique_predicted_datas = map(np.unique, np.array(predicted_datas).T)
    nb_of_el = list(map(np.size, unique_predicted_datas))
    pairs = enumerate(nb_of_el)
    idx, _ = zip(*sorted(pairs, key=lambda x:x[1], reverse=True)[:cutoff])
    idx = unlabeled_idx[list(idx)]
    labeled_idx = np.concatenate([labeled_idx, idx])
    return labeled_idx

def picker_2max(model, data, labeled_idx, cutoff):
    unlabeled_idx = np.setdiff1d(np.arange(data.shape[0]), labeled_idx)
    predicted_data = model.predict(data[unlabeled_idx])
    predicted_data.T[::-1].T.sort()
    two_max_dist = predicted_data.max(axis=1) - np.delete(predicted_data,[0], axis=1).max(axis=1)
    pairs = enumerate(two_max_dist)
    idx, _ = zip(*sorted(pairs, key=lambda x:x[1], reverse=True)[:cutoff])
    idx = unlabeled_idx[list(idx)]
    labeled_idx = np.concatenate([labeled_idx, idx])
    return labeled_idx