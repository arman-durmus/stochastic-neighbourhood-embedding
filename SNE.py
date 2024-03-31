import numpy as np
from artificial_data import generate_datasets
import matplotlib.pyplot as plt

def calculate_similarity(data, index, std):
    similarity = np.zeros(len(data))
    for j in range(len(data)):
        if index == j:
            # We only care about pairwise similarities between different points
            similarity[j] = 0
        else:
            new_sim = np.exp(-(np.linalg.norm(data[index] - data[j]) ** 2) / (2 * std ** 2)) / sum(
                np.exp(-(np.linalg.norm(data[index] - data[k]) ** 2) / (2 * std ** 2))
                for k in range(len(data))
                if k != index
            )
            similarity[j] = new_sim
    return similarity

def get_gradient(sim_X, sim_Y, Y, i):
    n = len(Y)
    return 2*sum((sim_X[i][j] + sim_X[j][i] - sim_Y[i][j] - sim_Y[j][i])*(Y[i]-Y[j]) for j in range(n))

def kullback_leibler(sim_x, sim_y):
    assert len(sim_x) == len(sim_y)
    
    kl_divergence = 0
    for i in range(len(sim_x)):
        if sim_x[i] != 0 and sim_y[i] != 0:
            kl_divergence += sim_x[i] * np.log(sim_x[i] / sim_y[i])
    
    return kl_divergence

def SNE(data, y_dim, num_iters, learning_rate):
    n = len(data)
    similarity_X = []
    std = np.zeros(n)
    for i in range(n):
        similarity_X.append(calculate_similarity(data, i, 10))
    similarity_X = np.array(similarity_X)
    # initialize Y
    Y = np.random.normal(loc=0, scale=0.0001, size=(n, y_dim))
    for iteration in range(num_iters):
        print("iteration ", iteration)
        # similarity_Y = np.array([calculate_similarity(Y, i, 1/np.sqrt(2)) for i in range(n)])
        similarity_Y = []
        for i in range(n):
            similarity_Y.append(calculate_similarity(Y, i, 1/np.sqrt(2)))
        similarity_Y = np.array(similarity_Y)

        gradient = []
        for i in range(n):
            gradient.append(get_gradient(similarity_X, similarity_Y, Y, i))
        gradient = np.array(gradient)
        print('Kullback-Leibler Divergence:', sum(kullback_leibler(similarity_X[i], similarity_Y[i]) for i in range(n)))
        Y = Y-learning_rate*gradient
    return Y

def tSNE(data, y_dim, num_iters, learning_rate):
    n = len(data)
    similarity_X = []
    std = np.zeros(n)
    for i in range(n):
        similarity_X.append(calculate_similarity(data, i, 10))
    similarity_X = np.array(similarity_X)
    similarity_X = np.add(similarity_X, np.transpose(similarity_X)) / (2 * n)
    print(sum(sum(similarity_X)))
    Y = np.random.normal(loc=0, scale=0.0001, size=(n, y_dim))
    for iteration in range(num_iters):
        # change this (y imilarities calculated differently)
        print("iteration ", iteration)
        # similarity_Y = np.array([calculate_similarity(Y, i, 1/np.sqrt(2)) for i in range(n)])
        similarity_Y = []
        for i in range(n):
            similarity_Y.append(calculate_similarity(Y, i, 1/np.sqrt(2)))
        similarity_Y = np.array(similarity_Y)
        print(similarity_Y)
        
        #print(kullback_leibler(similarity_X, similarity_Y))
        gradient = []
        for i in range(n):
            gradient.append(get_gradient(similarity_X, similarity_Y, Y, i))
        gradient = np.array(gradient)
        print(sum(kullback_leibler(similarity_X[i], similarity_Y[i]) for i in range(n)))
        Y = Y-learning_rate*gradient
    return Y

if __name__ == "__main__":
    noisy_circles_df, noisy_moons_df, blobs_df, aniso, varied_df = generate_datasets(50)
    blobs = np.array(blobs_df[["a", "b"]])
    Y = SNE(blobs, y_dim=1, num_iters=100, learning_rate=0.5)
    plt.figure()
    ax = plt.subplot()
    plt.scatter(Y.T[0], np.zeros(50), c=blobs_df.target)
    plt.title(f'tSNE Map')
    plt.xlabel('Dimension 1')
    plt.grid(True)
    plt.show()
