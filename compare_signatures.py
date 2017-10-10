import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


def compare_signatures(sig1, sig2):
    # sig1 is 30-by-96
    # sig2 is 27-by-96
    M = cosine_similarity(sig1, sig2)
    # print(M)
    plt.imshow(M, cmap='hot', interpolation='nearest')
    plt.xlabel("calculated signatures")
    plt.ylabel("calculated signatures")
    plt.title("Cosine Similarity of calculated Signatures")
    plt.colorbar()
    plt.show()
    # signature_names = np.genfromtxt("data/signatures/signatures.txt", dtype=str, usecols=list(range(3,30)), max_rows=1, delimiter='\t')    # M is 30-by-27 so for each of the 30 signatures in sig1, we get their cosine similarity to each of the 27 signatures in sig2
    # T = range(M.shape[0])
    # for i in range(M.shape[1]):
    #     print("For " + str(signature_names[i]))
    #     print("max cosine similarity is " + str(np.amax(M[:,i])))
    #     print("corresponding calculated signature is " + str(np.argmax(M[:,i])))
    #     plt.plot(T, M[:,i])
    # plt.show()

def compare_calculated_simulated_signatures():
    calculated_signatures = np.load("output/9_29_17/signatures/calculated-signatures-5.npy")
    actual_signatures = np.load("data/examples/example-signatures.npy")
    compare_signatures(calculated_signatures, actual_signatures)

def compare_calculated_cosmic_signatures():
    N = 27
    calculated_signatures = np.load("output/10_5_17/signatures/real-data-signatures-" + str(N) + ".npy")
    calculated_categories = np.loadtxt("data/whole_exome/ALL_exomes_mutational_catalog_96_subs.txt", dtype=str, skiprows=1, usecols=list(range(0,1)))

    cosmic_signatures = np.loadtxt("data/signatures/signatures.txt", dtype=np.dtype(np.float32), delimiter="\t", skiprows=1, usecols=list(range(3,30)))
    cosmic_signatures = cosmic_signatures.transpose()
    cosmic_categories = np.loadtxt("data/signatures/signatures.txt", dtype=str, skiprows=1, usecols=list(range(2,3)))
    cosmic_signatures = standardized_signature_category_order(calculated_categories, cosmic_signatures, cosmic_categories)
    # compare_signatures(cosmic_signatures, original_signatures)
    compare_signatures(calculated_signatures, calculated_signatures)

# given the order of categories in one group of signatures and the order of the categories in another group of signatures
# re-order the categories in the second group so they're the same order
def standardized_signature_category_order(categories1, sig2, categories2):
    l = list()
    for category in categories1:
        l.append(sig2[:, categories2 == category])
    return np.concatenate(l, axis=1)


if __name__ == '__main__':
    compare_calculated_cosmic_signatures()
    # compare_calculated_simulated_signatures()
