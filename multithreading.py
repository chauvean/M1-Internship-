from my_clustering import *
import multiprocessing
from time import time
import pandas as pd
from threading import Thread


def sil_and_cluster_threaded(timeserie, nb_clusters, result, i):
    """
    calls sil and cluster calculation and puts it in a buffer shared by all threads. ( common buffer not correctly implemented yet.)
    """
    result[i] = sil_and_cluster(timeserie, nb_clusters)


def choose_threaded(df, wanted_clusters):
    """
    uses threading library to parallelise silhouette score and cluster calculation
    creates list of runnable threads, runs and join them at the end.

    Problem : slower than without threads

    :param df:
    :param wanted_clusters:
    :return:
    """
    l = []
    array = get_infos(df, 'delta_T0', False)
    ndarray = [[el] for el in array]
    timeserie = to_time_series_dataset(ndarray)
    threads = [Thread() for _ in range(len(wanted_clusters))]
    results = [[] for _ in range(len(wanted_clusters))]
    for i in range(len(wanted_clusters)):
        threads[i] = Thread(target=sil_and_cluster_threaded, args=(timeserie, wanted_clusters[i], results, i))
        threads[i].start()
    for i in range(len(wanted_clusters)):
        threads[i].join()


def choose_multiprocess(df, wanted_clusters, procs):
    """
    same as choose_threaded but with multiprocessing which is there faster than serial.
    :param df: input frame with infos
    :param wanted_clusters: list of number of clusters we want to test.
    :param procs: nbr of process to parallelise. ( mostly 1 per wanted cluster )
    :return:
    """
    results = [1 for _ in range(len(wanted_clusters))]
    jobs = []
    array = get_infos(df, 'delta_T0', False)
    ndarray = [[el] for el in array]
    timeserie = to_time_series_dataset(ndarray)
    for i in range(0, procs):
        process = multiprocessing.Process(target=sil_and_cluster_threaded,
                                          args=(timeserie, wanted_clusters[i], results, i))
        jobs.append(process)

    for j in jobs:
        j.start()

    # Ensure all of the processes have finished
    for j in jobs:
        j.join()


def draw_cluster_coincidence(df, wanted_clusters, step=1, title="", option="show", output_file=""):
    df_list = []
    X = []
    Y = []
    n_samples = len(df['delta_T0']) - 2
    for index in range(2, n_samples, step):
        df_list.append(df.iloc[:index])
    for index, frame in enumerate(df_list):
        t1 = time()
        choose_multiprocess(frame, wanted_clusters, len(wanted_clusters))
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


if __name__ == "__main__":
    # procs  = 3
    # t1 = time()
    # dataf = pd.read_excel('bio_coincide_full3.xlsx')
    # tint = time()
    # print(tint - t1)
    # dataf = dataf.sample(frac=1).reset_index(drop=True)
    # w = [2, 3, 4]
    # draw_cluster_coincidence(dataf[:10000], w, 1000, "", "savefig", "cluster_ndcoincidence.png")
    # # choose_serial(df[:1000], wanted_clusters)
    # # choose_multiprocess(dataf[:], w, procs)
    # t2 = time()
    # print("total time : {}".format(t2 - tint))
    # x = [1000 * i for i in range(1, 8)]
    # x.append(9000)
    # y = [27, 110, 240, 420, 660, 960, 1250, 5800]
    # plt.plot(x, y)
    # plt.savefig("cluster_coincidence_complexity2")
    pass
