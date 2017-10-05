import numpy as np
import os
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import nnls

import matplotlib.pyplot as plt

# run kmeans on X for N clusters
def run_kmeans(X, N):
    kmeans = KMeans(n_clusters=N)
    kmeans.fit_predict(X)
    return kmeans

# eliminate any columns that together add up to less than 1% of the total mutations
def reduce_dimension(M):
    fraction_per_category = M.sum(axis=0)/M.sum()
    fraction_per_category.sort()
    for i in range(1, 97):
        # find the minimum category
        if sum(fraction_per_category[:i]) > .01:
            # we want to remove the smallest i-1 categories
            vals_to_remove = fraction_per_category[:i-1]
            cols_to_remove = np.isin(M.sum(axis=0)/M.sum(), vals_to_remove)
            return np.logical_not(cols_to_remove)


# for a given n, the number of signatures, calculate the signatures from M
def calculate_signatures(N, M):
    # M is a numpy matrix where rows represent samples and columns represent mutation categories
    # cols_to_keep = reduce_dimension(M)
    # M = M[:, cols_to_keep]
    silhouette_score_list = []
    reconstruction_error = []
    G, K = M.shape
    P_list = [] # signature_matrix_list
    # For I iterations (usually 400 < I < 500)
    num_iterations = 0
    while num_iterations < 1001:
        print(num_iterations)
        P = run_iteration(N, M)
        P_list.append(P)
        num_iterations = num_iterations + 1
        if (num_iterations % 10 == 0):
            # calculate the iteration averaged matrix and see if it has converged
            P = np.concatenate(P_list, axis=0)
            # run k-means clustering on all signatures
            P_kmeans = run_kmeans(P, N)
            # P_kmeans.cluster_centers_ returns the centers of the clusters
            # kmeans.labels_ returns the cluster that each point belongs to
            s_score = silhouette_score(X=P, labels=P_kmeans.labels_, metric='cosine')
            silhouette_score_list.append(s_score)
            print("silhouette score: " + str(s_score))
            # calculate Frobenius reconstruction errors
            error = 0
            # E_list = [] # exposure_matrix_list
            # given the set of N signatures, estimate the exposure to each for each sample
            for sample in range(0, G):
                _, rnorm = nnls(P_kmeans.cluster_centers_.transpose(), M[sample, :].transpose())
                error += rnorm
            reconstruction_error.append(error)
            print("Frobenius reconstruction error: " + str(error))

    with open("output/10_5_17/metrics/silhouette_score.txt", "w") as output:
        output.write(str(silhouette_score_list))
    with open("output/10_5_17/metrics/reconstruction_error.txt", "w") as output:
        output.write(str(reconstruction_error))
    P = np.concatenate(P_list, axis=0)
    P_kmeans = run_kmeans(P, N)
    return P_kmeans.cluster_centers_

def run_iteration(N, M):
    G, K = M.shape
    # run Monte Carlo bootstrap resampling
    rows_to_keep = np.random.choice(G, G)
    M_prime = M[rows_to_keep,]
    # run NMF until convergence (10,000 runs without change) or until reach max num runs (1,000,000 total runs)
    model = NMF(n_components=N, init='random', solver='mu', max_iter=1000000)
    W = model.fit_transform(M_prime)
    H = model.components_
    return H

def run_synthetic_data():
    # TODO - loop through multiple values of N
    N = 5
    M = np.loadtxt("data/examples/example-mutation-counts.tsv", dtype=np.dtype(np.int32), delimiter="\t", skiprows=1, usecols=list(range(1,21)))
    signatures = calculate_signatures(N, M)
    np.save("output/9_29_17/signatures/synthetic-signatures-" + str(N) + ".npy", signatures)
    actual_signatures = np.load("data/examples/example-signatures.npy")
    # smaller pairwise distance means more similar
    # M = cosine_similarity(actual_signatures, signatures)
    # T = range(M.shape[0])
    # for i in range(M.shape[1]):
    #     plt.plot(T, M[:,i])
    # plt.show()

def get_real_data():
    # define a generator that strips the first column
    def strip_first_col(fname, delimiter=None):
        with open(fname, 'r') as fin:
            for line in fin:
                try:
                    yield line.split(delimiter, 1)[1]
                except IndexError:
                    continue
    M_list = []
    exome_files = os.listdir("data/whole_exome/")
    for exome_file in exome_files:
        fname = "data/whole_exome/" + exome_file
        # print(fname)
        M = np.loadtxt(strip_first_col(fname), dtype=np.dtype(np.int32), delimiter="\t", skiprows=1, ndmin=2)
        M = M.transpose()
        # print(M.shape)
        M_list.append(M)
        # now rows in M represent samples and columns in M represent mutational categories
    genome_files = os.listdir("data/whole_genome/")
    for genome_file in genome_files:
        fname = "data/whole_genome/" + genome_file
        # print(fname)
        M = np.loadtxt(strip_first_col(fname), dtype=np.dtype(np.int32), delimiter="\t", skiprows=1, ndmin=2)
        M = M.transpose()
        # print(M.shape)
        M_list.append(M)
        # now rows in M represent samples and columns in M represent mutational categories
    M = np.concatenate(M_list)
    return(M)

def run_real_data():
    N_list = list(range(27, 28))
    for N in N_list:
        print(N)
        M = get_real_data()
        signatures = calculate_signatures(N, M)
        np.save("output/10_5_17/signatures/real-data-signatures-" + str(N) + ".npy", signatures)


if __name__ == '__main__':
    # run_synthetic_data()
    run_real_data()
