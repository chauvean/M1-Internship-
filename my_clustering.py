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
    # ti = time()
    # # # X = to_time_series_dataset([[1], [1], [1], [5], [50], [51], [52], [53], [200], [200.1], [200.2]])
    # # # km = TimeSeriesKMeans(n_clusters=4, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(X)
    # # t2 = time()
    # # #timelapses = choose_timelapses('bio_ precede_full3.xlsx')
    # # cluster, sil_score = choose_timelapses('test.xlsx')
    # # # cluster, sil_score = choose_timelapses('bio_precede_full3.xlsx')
    # # t3 = time()
    # # print(t3 - t2)
    # # t1 = time()
    # in_df = pd.read_excel('bio_coincide_full3.xlsx')
    # df = in_df[:150]
    # # t1 = time()
    # #l = pattern_for_each_cluster_coin(df)
    # choose_timelapses(df)
    # # l = n_sized_coincidence(df)
    # # t2 = time()
    # # print("draw_time = {}".format(t2-t1))
    # # df = pd.read_excel('bio_precede_full3.xlsx')
    # # df = pd.read_excel('test.xlsx')
    # # best_clustering = choose_timelapses(df)[0]
    # # filled_clusters = associate_traj_cluster(df, best_clustering)
    # # df1 = pd.read_excel('test.xlsx')
    # # seqs = pattern_for_each_cluster_prec('test.xlsx')
    # # data = [['hey', 1], ['aaa', 2]]
    # # s = pd.Series(data, index=['delta_T0', 'b'])
    # # a = s['b']
    # # date = date_coin(s)
    # # data = [['hey' for _ in range(15)]]
    # # df = pd.DataFrame(data)
    # # df.columns = df1.columns
    # # t1 = time()
    # # df = pd.read_excel('coincide_test.xlsx')
    # # all_seqs = pattern_for_each_cluster_coin(df)
    # # long_seqs = []
    # # for el in all_seqs:
    # #     long_seqs.append(select_collisions(el))
    # # t2 = time()
    # # print(" find coincidences in {} for {} lines ".format(t2 - t1, len(df['Source'])))
    # # to_excel_complexity('coincide_test.xlsx', 'coincidences_times.xlsx', 10, 2, 100)
    # # t2 = time()
    # # print(t2-ti)
    # # to_excel_complexity('coincide_test.xlsx', 'coincidences_times.xlsx', 10, 300, 400)
    # # t1 = time()
    # # draw_coincidence_complexity('coincide_test.xlsx', 20)
    # # t2 = time()
    # print(t2-t1)
    # ti = time()
    # t1 = time()
    #df = pd.read_excel('bio_coincide_full3.xlsx')

    # t2 = time()
    # print("time to read file : {}".format(t2 - t1))
    # # # d = df[:100]
    # # # t1 = time()
    # # # draw_sil_complexity(df[:10000])
    # # # # print("read_excel time : {}".format(t1 - ti))
    # # # # output_file = "test.png"
    # # # # choose_timelapses(df, 2, 40000)
    b = load_workbook('test.xlsx')

    # for size in range(2, len(df['delta_TO'])):
    #     array = get_infos(df[:size], 'delta_T0')
    #     ndarray = [[el] for el in array]
    #     timeserie = to_time_series_dataset(ndarray)
    #     t1 = time()
    #     TimeSeriesKMeans(n_clusters=3, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(
    #         timeserie)
    #     t2 = time()
    # draw_clustering_complexity(df, [2, 3, 4, 5], 200, "", "savefig", "clustering_complexity.png")
    # # # tf = time()
    # # # print("draw_cluster : {}".format(tf - t1))
    # # """
    # # modify size of samples
    # # """
    # # # x = []
    # # # c = 0
    # # # for longueur in range(100, 20000, 1000):
    # # #     array = get_infos(df[:longueur], 'delta_T0')
    # # #     ndarray = [[el] for el in array]
    # # #     timeserie = to_time_series_dataset(ndarray)
    # # #     cluster = TimeSeriesKMeans(n_clusters=4, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(
    # # #         timeserie)
    # # #     ta = time()
    # # #     silhouette_score(timeserie, cluster.labels_)
    # # #     tb = time()
    # # #     print(tb - ta)
    # # #     c += 1
    # # #     x.append(tb - ta)
    # # # y = range(c)
    # # # plt.xlabel("number of samples")
    # # # plt.ylabel("silouhette score calculation")
    # # # plt.plot(x, y)
    # # # plt.show()
    # # # """
    # # # modify n clusters
    # # # """
    # # # x = []
    # # # y = []
    # # # c = 0
    # # # array = get_infos(df[:500], 'delta_T0', False)
    # # # ndarray = [[el] for el in array]
    # # # timeserie = to_time_series_dataset(ndarray)
    # # # for n_cluster in range(2, len(array), 100):
    # # #     ta = time()
    # # #     cluster = TimeSeriesKMeans(n_clusters=n_cluster, verbose=False, max_iter=5, metric="dtw", random_state=0).fit(
    # # #         timeserie)
    # # #     tb = time()
    # # #     # silhouette_score(timeserie, cluster.labels_)
    # # #     print(tb - ta)
    # # #     y.append(tb - ta)
    # # #     x.append(n_cluster)
    # # # plt.xlabel("number of samples")
    # # # plt.ylabel("kmeans complexity")
    # # # plt.plot(x, y)
    # # # plt.savefig('kmeans_calc3.png')
    # # # t1 = time()
    # # choose_timelapses(df[:1000], 10, 2)
    # # # t2 = time()
    # # # print(" not threaded version : {}".format(t2 - t1))
    # # tb = time()
    # # choose_timelapses_threaded(df[:1000], 500)
    # # ta = time()
    # # print("threaded version : {}".format(ta - tb))
    # ti = time()
    # choose_serial(df[:200], [3, 4, 5, 6])
    # tinter = time()
    # print("unthreaded : {}".format(tinter-ti))
    # choose_threaded(df[:200], [3, 4, 5, 6])
    # tf = time()
    # print("total time : {}".format(tf - ti))
    # l = [['a', 1], ['c', 5], ['b', 3]]
    # l.sort(key=lambda x: x[1])
    # print(l[-1])
    # df = pd.read_excel('test.xlsx')
    # t1 = time()
    # print(choose_serial(df[:1000], [2, 3]))
    # t2 = time()
    # print()
    print("test passed")


if __name__ == "main":
    main()
