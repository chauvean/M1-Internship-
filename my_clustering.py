from clustering import *
from useful_functions import *
from matplotlib import pyplot as plt
import numpy as np
from time import time
from openpyxl import load_workbook
import pandas as pd

def associate_traj_cluster(df, best_cluster, arg):
    """
    kmeans gave us the clusters . But we cant access all infos from the cluster members with what kmeans returns.
    We associate each samples with it's label manually.
    """
    if best_cluster != 1:
        filled_clusters = [[] for _ in range(best_cluster.n_clusters)]
        for index_traj, traj in df.iterrows():
            cluster_index = np.where(best_cluster.X_fit_ == traj[arg])  # gets all indexes verifying the property
            filled_clusters[best_cluster.labels_[cluster_index[0][0]]].append(traj)
        return filled_clusters
    else:
        return 0


def _check_no_empty_cluster(labels, n_clusters):
    for k in range(n_clusters):
        if numpy.sum(labels == k) == 0:
            raise EmptyClusterError
    else:
        return 1


def draw_clustering_complexity(df, wanted_cluster, step=1, title="", option="show",
                               output_file=""):
    df_list = []
    X = []
    Y = []

    n_tries = len(df['delta_T0']) - 2
    for index in range(2, n_tries, step):
        df_list.append(df.iloc[:index])
    for index, frame in enumerate(df_list):
        t1 = time()
        choose_serial(frame, wanted_cluster)
        t2 = time()
        nb_samples = len(frame['delta_T0'])
        X.append(nb_samples)
        Y.append(t2 - t1)
    plt.plot(X, Y)
    plt.title(title)
    plt.xlabel('number of samples')
    plt.ylabel('complexity (seconds)')
    if option == "show":
        plt.show()
    elif option == "savefig":
        plt.savefig(output_file)
    else:
        print("Invalid option")
        return 1


def draw_sil_complexity(df):
    y = []
    c = 0
    array = get_infos(df, 'delta_T0')
    ndarray = [[el] for el in array]
    nb_samples = len(ndarray)
    timeserie = to_time_series_dataset(ndarray)
    for nb_clusters in range(2, nb_samples, 100):
        print(nb_clusters)
        cluster = TimeSeriesKMeans(n_clusters=nb_clusters, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(
            timeserie)
        ta = time()
        silhouette_score(timeserie, cluster.labels_)
        tb = time()
        y.append(tb - ta)
        c += 1
    plt.plot(range(c), y)
    plt.show()


def sil_and_cluster(timeserie, nb_clusters):
    t1 = time()
    cluster = TimeSeriesKMeans(n_clusters=nb_clusters, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(
        timeserie)
    given_labels = []
    for label in cluster.labels_:
        if label not in given_labels:
            given_labels.append(label)
    if len(given_labels) > 1 and _check_no_empty_cluster(cluster.labels_, nb_clusters):
        sil = silhouette_score(timeserie, cluster.labels_)
        t2 = time()
        print("time to cluster and calc sil : {}".format(t2 - t1))
        print("sil : {}".format(sil))
        return cluster, sil
    else:
        return 1, 0


def choose_serial(df, wanted_clusters):
    l = []
    array = get_infos(df, 'delta_T0', False)
    ndarray = [[el] for el in array]
    timeserie = to_time_series_dataset(ndarray)
    for nb_cluster in wanted_clusters:
        l.append([sil_and_cluster(timeserie, nb_cluster)])
    l.sort(key=lambda x: x[0][1])
    return l[-1]


def main():

    

    df = pd.read_excel('test.xlsx')
    df = pd.read_excel('bio_coincide_full3.xlsx')
    t1 = time()
    choose_threaded(df[:200], [3, 4, 5, 6])
    choose_serial(df[:1000], [2, 3])
    t2 = time()
    print("total time : {}".format(t2-t1)
    print("test passed")


if __name__ == "main":
    main()
