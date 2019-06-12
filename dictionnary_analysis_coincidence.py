from my_clustering import *


def pattern_for_each_cluster_coin(df, wanted_clusters):
    """
    :return: list of dictionnaries, with all coincidences for each cluster
    """
    best_clustering = choose_serial(df, wanted_clusters)[0]
    filled_clusters = associate_traj_cluster(df, best_clustering, 'delta_T0')
    if filled_clusters != 0:
        df_filled_clusters = list_serie_to_df(filled_clusters)
        for filled_df in df_filled_clusters:
            filled_df.columns = df.columns
        sequences = []
        for cluster in df_filled_clusters:
            sequences.append(select_collisions(n_sized_coincidence(cluster)))
        return sequences
    else:
        return 0


def n_sized_coincidence(df):
    """
    :return: dictionnary where key is a pattern ( ex: (HGB_L -> BLA_N )
    and values are all the possible dates and pat_did's in the samples
    """
    dic = dict()
    for pat_info_row in df.iterrows():
        if (pat_info_row[1]['Source'], pat_info_row[1]['Target']) in dic.keys():
            dic[(pat_info_row[1]['Source'], pat_info_row[1]['Target'])].append([pat_info_row[1]['date'],
                                                                                pat_info_row[1]['delta_T0'],
                                                                                pat_info_row[1]['trajID']]
                                                                               )
        else:
            dic[(pat_info_row[1]['Source'], pat_info_row[1]['Target'])] = [[pat_info_row[1]['date'],
                                                                            pat_info_row[1]['delta_T0'],
                                                                            pat_info_row[1]['trajID']]]
        pass
    return dic


def draw_coincidence_dic_complexity(df, wanted_cluster, step=1, title="", option="show",
                                    output_file=""):
    df_list = []
    X = []
    Y = []
    n_samples = len(df['delta_T0']) - 2
    for index in range(2, n_samples, step):
        df_list.append(df.iloc[:index])
    for index, frame in enumerate(df_list):
        t1 = time()
        select_collisions(n_sized_coincidence(frame))
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


def main():
    t1 = time()

    # choose excel file to extract data from ( file must be in the same directory as the running code
    excel_file = 'test.xlsx'
    df = pd.read_excel(excel_file)
    t2 = time()
    print("open data time : {}".format(t2 - t1))
    # filter interesting columns (spped up a little bit)
    df2 = df[['Source', 'Target', 'trajID', 'delta_T0']]
    
    #transform into tslearn functions understandable format
    array = get_infos(df[:1000], 'delta_T0', False)
    ndarray = [[el] for el in array]
    timeserie = to_time_series_dataset(ndarray)
    
    
    #parameters
    wanted_clusters = [2, 3, 4, 5]
    step = 1000 #draw_coinc only execute for k*step number of samples
    option = "savefig"
    output_file = "coincidence_dic_complexity.png"
    
    #uncomment the following tests
    # sil_and_cluster(timeserie, 2)
    # draw_coincidence_dic_complexity(df, , step , "", option, output_file)

    t3 = time()
    print("total time : {}".format(t3 - t2))

if __name__ == 'main':
    main()
