import pandas as pd


def select_collisions(dic, n=1):
    """
    :return: dictionnary where keys with less than n values are suppressed.
    """
    return {k: v for k, v in dic.items() if len(v) > n}


def get_infos(buff, column, distinct=True):
    l = []
    if type(buff) == 'list':
        for ser in buff:
            if ser.values[0] not in l:
                l.append(ser.values[0])
    else:
        if distinct:
            for date in buff[column]:
                if date not in l:
                    l.append(date)
        else:
            for date in buff[column]:
                l.append(date)
    return l


def get_infos_multiple_columns(df, columns):
    """
    same as get_infos when we want values of multiple columns
    :return: l with values of required columns
    """
    l = []
    for column in columns:
        l.append(get_infos(df, column))
    return l


def get_column(df, column, value):
    return df[df[column] == value]


def filter_df_by_date(df, col_name):
    df_list = []
    for date in get_infos(df, col_name):
        df_list.append(df[df[col_name] == date])
    return df_list


def filter_excel_by_column(excel_file, column):
    df = pd.read_excel(excel_file)
    frames = [get_column(df, column, value) for value in get_infos(df, column)]
    return frames


def date_to_str(date):
    y, m, d = date.year, date.month, date.day
    if len(str(m)) == 1:
        return ("-").join([str(y), '0' + str(m), str(d)])
    else:
        return ("-").join([str(y), str(m), str(d)])


def frequences(dict):
    return [[key, len(dict[key])] for key in dict.keys]


def list_serie_to_df(series_list):
    """
    in associate_traj_cluster we use iterrows which return series
    we convert to df as other functions needs df instead of series
    :return: dataframe of the list of series
    """
    df_list = []
    for series in series_list:
        df_for_one_cluster = []
        for serie in series:
            df_for_one_cluster.append(pd.DataFrame([list(serie.values)]))
        df_list.append(pd.concat(df_for_one_cluster))
    return df_list
