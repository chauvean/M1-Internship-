from useful_functions import *
from time import time
import itertools
from my_clustering import *
from useful_functions import *
import datetime


class Date_prec:
    def __init__(self, serie=None, df=None):
        self.content = []
        self.source = []
        self.target = []
        if df is not None:
            self.date_from = df.at[df.first_valid_index(), 'datefrom']
            self.date_to = df.at[df.first_valid_index(), 'dateto']
            self.delta_T0 = df.at[df.first_valid_index(), 'delta_T0']
            for source, target, deltat0 in zip(df['Source'], df['Target'], df['delta_T0']):
                self.content.append([source, target])
                self.source.append(source)
                self.target.append(target)
                self.delta_T0 = deltat0
                pass
        if serie is not None:
            self.delta_T0 = 0
            pass
        self.tuples = None

    def set_nb_tuples(self, nb_combi):
        self.tuples = [[[], []] for _ in range(nb_combi)]


class Sequence:
    def __init__(self, date1, date2, content=[]):
        self.patdid = -1
        self.datefrom = date1
        self.dateto = date2
        self.content = content


def n_sized_precedence(df, n, col_name):
    #     if col_name == 'delta_T0':
    #         dfs_list = filter_df_by_date(df, col_name)
    #     if col_name == 'source-target':
    #         pass
    dfs_list = filter_df_by_date(df, col_name)
    class_list = []
    for df in dfs_list:
        class_list.append(Date_prec(None, df))
    for date_class in class_list:
        if n == 1:
            list_of_tuples = date_class.content
        else:
            list_of_tuples = list(itertools.combinations(date_class.content, n))
        date_class.set_nb_tuples(len(list_of_tuples))
        for tuple_index, tuple_sample in enumerate(list_of_tuples):
            for i in range(n):
                date_class.tuples[tuple_index][0].append(tuple_sample[i])
                date_class.tuples[tuple_index][1].append(tuple_sample[i])
                # when n > 1 we should use the next syntax as we will not have [ a, b] but [{a,b],[c,d]] as we are
                # talking about combinations and not single evaluations
                # date_class.tuples[tuple_index][0].append(tuple_sample[i][0])
                # date_class.tuples[tuple_index][1].append(tuple_sample[i][1])
    return class_list


def find_seq_by_deltat0(str_t1, str_t2, dic):
    """
    works if delta_t0   are no holes in the indexes. ( 0,1,2,3,4 is ok 0,2,3,4 isn't)
    I check the next delta_T0, range 1 then 2 then 3. But delta_T0 sequences will have holes after clustering.
    :param str_t1:
    :param str_t2:
    :param dic:
    :return:
    """
    seq_list = []
    t1 = int(str_t1)
    t2 = int(str_t2)
    for seq_index, seq in enumerate(dic[t1]):
        seq_list.append(Sequence(t1, t2, [seq[0][0], seq[1][0]]))
    t = t1
    smaller_sized_seqs = []
    while t < t2:
        new_seqs = []
        for seq in seq_list:
            if date_to_str(t + 1) in dic.keys():
                for pair in dic[date_to_str(t + 1)]:
                    if pair[0][0] == seq.content[-1]:
                        new_seqs.append(Sequence(seq.datefrom, seq.dateto, seq.content + [pair[1][0]]))
        seq_list = new_seqs
        t += 1
    for el in smaller_sized_seqs:
        seq_list.append(el)
    return seq_list


def find_seq_chained_lists(str_date1, str_date2, dic):
    year, month, day = str_date1.split("-")
    date1 = datetime.datetime(int(year), int(month), int(day))
    year, month, day = str_date2.split("-")
    date2 = datetime.datetime(int(year), int(month), int(day))
    seq_list = []
    for seq_index, seq in enumerate(dic[str_date1]):
        seq_list.append(Sequence(date1, date2, [seq[0][0], seq[1][0]]))

    date = date1
    smaller_sized_seqs = []
    while date < date2:
        new_seqs = []
        for seq in seq_list:
            if date_to_str(date + datetime.timedelta(days=1)) in dic.keys():
                for pair in dic[date_to_str(date + datetime.timedelta(days=1))]:
                    if pair[0][0] == seq.content[-1]:
                        new_seqs.append(Sequence(seq.datefrom, seq.dateto, seq.content + [pair[1][0]]))
                    else:
                        smaller_sized_seqs.append(seq)
        seq_list = new_seqs
        date += datetime.timedelta(days=1)
    for el in smaller_sized_seqs:
        seq_list.append(el)
    return seq_list


def fill_dic(class_list, arg):
    dic = dict()
    for seq_class in class_list:
        if arg == 'datefrom':
            if seq_class.date_from in dic.keys():
                dic[seq_class.date_from].append(seq_class.tuples)
            else:
                dic[seq_class.date_from] = seq_class.tuples
            print("ok")
        if arg == 'delta_T0':
            if seq_class.delta_T0 in dic.keys():
                dic[seq_class.delta_T0].append(seq_class.tuples)
            else:
                dic[seq_class.delta_T0] = seq_class.tuples
        if arg == 'source-target':
            if seq_class.sourcetarget in dic.keys():
                dic[seq_class.sourcetarget].append(seq_class.tuples)
            else:
                dic[seq_class.delta_T0] = seq_class.tuples
    return dic


def pattern_for_each_cluster_prec(excel_file):
    df = pd.read_excel(excel_file)
    best_clustering = choose_serial(df)[0]
    filled_clusters = associate_traj_cluster(df, best_clustering)
    df_filled_clusters = list_serie_to_df(filled_clusters)
    for filled_df in df_filled_clusters:
        filled_df.columns = df.columns
    l = get_infos(pd.read_excel(excel_file), 'delta_T0')
    l.sort()
    sequences = []
    for cluster in df_filled_clusters:
        dic = fill_dic(n_sized_precedence(cluster, 1, 'delta_T0'))
        sequences.append(find_seq_by_deltat0(l[0], l[-1], fill_dic(n_sized_precedence(cluster, 1, 'delta_T0'))))

    return sequences


def main():
    t1 = time()
    df = pd.read_excel('bio_precede_full3.xlsx')

    t2 = time()
    print(t2 - t1)
    l2 = find_seq_chained_lists(l[0], l[-1], fill_dic(n_sized_precedence(df, 1, "datefrom"), "datefrom"))
    t3 = time()
    print(t3 - t2)


main()
