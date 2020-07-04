"""
"""
from numbers import Real
from typing import Union, Optional, List, Tuple, NoReturn

import numpy as np


__all__ = [
    "dict_to_str",
    "get_optimal_covering",
]


EMPTY_SET = []
Interval = Union[List[Real], Tuple[Real], type(EMPTY_SET)]
GeneralizedInterval = Union[List[Interval], Tuple[Interval], type(EMPTY_SET)]


def get_optimal_covering(total_interval:Interval, to_cover:list, min_len:int, split_threshold:int, traceback:bool=False, **kwargs) -> Tuple[GeneralizedInterval,list]:
    """ finished, checked,

    获取覆盖to_cover中每一项的满足min_len, split_threshold条件的最佳覆盖

    Parameters:
    -----------
    total_interval: 总的大区间
    to_cover: 需要覆盖的点和区间的列表
    min_len: 每一个覆盖的最小长度
    split_threshold: 覆盖之间的最小距离
    traceback: 是否记录每个covering覆盖了的to_cover的项（的index）
    注意单位保持一致！
    如果to_cover的范围超过total_interval的范围，会抛出异常

    Returns:
    --------
    (ret, ret_traceback)
        ret是一个GeneralizedInterval，满足min_len, split_threshold的条件；
        ret_traceback是一个list，
        其中每一项是一个list，记录了ret中对应的interval覆盖的to_cover中的项的indices
    """
    start_time = time.time()
    verbose = kwargs.get('verbose', 0)
    tmp = sorted(total_interval)
    tot_start, tot_end = tmp[0], tmp[-1]

    if verbose >= 1:
        print('total_interval =', total_interval, 'with_length =', tot_end-tot_start)

    if tot_end - tot_start < min_len:
        ret = [[tot_start, tot_end]]
        ret_traceback = [list(range(len(to_cover)))] if traceback else []
        return ret, ret_traceback
    to_cover_intervals = []
    for item in to_cover:
        if isinstance(item, list):
            to_cover_intervals.append(item)
        else:
            to_cover_intervals.append([item, item])
    if traceback:
        replica_for_traceback = deepcopy(to_cover_intervals)

    if verbose >= 2:
        print('to_cover_intervals after all converted to intervals', to_cover_intervals)

        # elif isinstance(item, int):
        #     to_cover_intervals.append([item, item+1])
        # else:
        #     raise ValueError("{0} is not an integer or an interval".format(item))
    # to_cover_intervals = interval_union(to_cover_intervals)

    for interval in to_cover_intervals:
        interval.sort()
    
    interval_sort_key = lambda i: i[0]
    to_cover_intervals.sort(key=interval_sort_key)

    if verbose >= 2:
        print('to_cover_intervals after sorted', to_cover_intervals)

    # if to_cover_intervals[0][0] < tot_start or to_cover_intervals[-1][-1] > tot_end:
    #     raise IndexError("some item in to_cover list exceeds the range of total_interval")
    # these cases now seen normal, and treated as follows:
    for item in to_cover_intervals:
        item[0] = max(item[0], tot_start)
        item[-1] = min(item[-1], tot_end)
    # to_cover_intervals = [item for item in to_cover_intervals if item[-1] > item[0]]

    # 确保第一个区间的末尾到tot_start的距离不低于min_len
    to_cover_intervals[0][-1] = max(to_cover_intervals[0][-1], tot_start + min_len)
    # 确保最后一个区间的起始到tot_end的距离不低于min_len
    to_cover_intervals[-1][0] = min(to_cover_intervals[-1][0], tot_end - min_len)

    if verbose >= 2:
        print('to_cover_intervals after two tails adjusted', to_cover_intervals)

    # 将间隔（有可能是负的，即有重叠）小于split_threshold的区间合并
    merge_flag = True
    while merge_flag:
        merge_flag = False
        new_intervals = []
        if len(to_cover_intervals) == 1:
            break
        for idx, item in enumerate(to_cover_intervals[:-1]):
            this_start, this_end = item
            next_start, next_end = to_cover_intervals[idx + 1]
            if next_start - this_end >= split_threshold:
                if split_threshold == (next_start - next_end) == 0 or split_threshold == (this_start - this_end) == 0:
                    # 需要单独处理 split_threshold ==0 以及正好有连着的单点集这种情况
                    new_intervals.append([this_start, max(this_end, next_end)])
                    new_intervals += to_cover_intervals[idx + 2:]
                    merge_flag = True
                    to_cover_intervals = new_intervals
                    break
                else:
                    new_intervals.append([this_start, this_end])
                    if idx == len(to_cover_intervals) - 2:
                        new_intervals.append([next_start, next_end])
            else:
                new_intervals.append([this_start, max(this_end, next_end)])
                new_intervals += to_cover_intervals[idx + 2:]
                merge_flag = True
                to_cover_intervals = new_intervals
                break
    if verbose >= 2:
        print('to_cover_intervals after merging intervals whose gaps < split_threshold', to_cover_intervals)

    # 此时，to_cover_intervals中所有区间的间隔都大于split_threshold
    # 但是除了头尾两个区间之外的区间的长度可能小于min_len
    ret = []
    ret_traceback = []
    if len(to_cover_intervals) == 1:
        # 注意，此时to_cover_intervals只有一个，这个元素（区间）的长度应该不小于min_len
        # 保险起见还是计算一下
        mid_pt = (to_cover_intervals[0][0]+to_cover_intervals[0][-1]) // 2
        half_len = min_len // 2
        if mid_pt - tot_start < half_len:
            ret_start = tot_start
            ret_end = min(tot_end, max(tot_start+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]
        else:
            ret_start = max(tot_start, min(to_cover_intervals[0][0], mid_pt-half_len))
            ret_end = min(tot_end, max(mid_pt-half_len+min_len, to_cover_intervals[0][-1]))
            ret = [[ret_start, ret_end]]

    start = min(to_cover_intervals[0][0], to_cover_intervals[0][-1]-min_len)

    for idx, item in enumerate(to_cover_intervals[:-1]):
        # print('item', item)
        this_start, this_end = item
        next_start, next_end = to_cover_intervals[idx + 1]
        potential_end = max(this_end, start + min_len)
        # print('start', start)
        # print('potential_end', potential_end)
        # 如果potential_end到next_start的间隔不够长，
        # 则进入下一循环（如果不到to_cover_intervals尾部）
        if next_start - potential_end < split_threshold:
            if idx < len(to_cover_intervals) - 2:
                continue
            else:
                # 此时 idx==len(to_cover_intervals)-2
                # next_start (从而start也是) 到 tot_end 距离至少为min_len
                ret.append([start, max(start + min_len, next_end)])
        else:
            ret.append([start, potential_end])
            start = next_start
            if idx == len(to_cover_intervals) - 2:
                ret.append([next_start, max(next_start + min_len, next_end)])
        # print('ret', ret)
    if traceback:
        for item in ret:
            record = []
            for idx, item_prime in enumerate(replica_for_traceback):
                itc = intervals_intersection([item, item_prime])
                len_itc = itc[-1] - itc[0] if len(itc) > 0 else -1
                if len_itc > 0 or (len_itc == 0 and item_prime[-1] - item_prime[0] == 0):
                    record.append(idx)
            ret_traceback.append(record)
    
    if verbose >= 1:
        print('the final result of get_optimal_covering is ret = {0}, ret_traceback = {1}, the whole process used {2} second(s)'.format(ret, ret_traceback, time.time()-start_time))
    
    return ret, ret_traceback


def dict_to_str(d:Union[dict, list, tuple], current_depth:int=1, indent_spaces:int=4) -> str:
    """
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = f"{{}}" if isinstance(d, dict) else f"[]"
        return s
    s = "\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    if isinstance(d, (list, tuple)):
        for v in d:
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{dict_to_str(v, current_depth+1)}\n"
            else:
                val = f'\042{v}\042' if isinstance(v, str) else v
                s += f"{prefix}{val}\n"
    elif isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{k}: {dict_to_str(v, current_depth+1)}\n"
            else:
                key = f'\042{k}\042' if isinstance(k, str) else k
                val = f'\042{v}\042' if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}\n"
    s += unit_indent*(current_depth-1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def plot_single_lead_ecg(s:np.ndarray, fs:Real, use_idx:bool=False, **kwargs) -> NoReturn:
    """ not finished

    single lead ECG plot,

    Parameters:
    -----------
    s: array_like,
        the single lead ECG signal
    fs: real,
        sampling frequency of `s`
    use_idx: bool, default False,
        use idx instead of time for the x-axis

    contributors: Jeethan, and WEN Hao
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    default_fig_sz = 120
    line_len = fs * 25  # 25 seconds
    nb_lines, residue = divmod(len(s), line_len)
    if residue > 0:
        nb_lines += 1
    for idx in range(nb_lines):
        idx_start = idx*line_len
        idx_end = min((idx+1)*line_len, len(s))
        c = s[idx_start:idx_end]
        secs = np.arange(idx_start, idx_end)
        if not use_idx:
            secs = secs / fs
        mvs = np.array(c) * 0.001
        fig_sz = int(round(default_fig_sz * (idx_end-idx_start)/line_len))
        fig, ax = plt.subplots(figsize=(fig_sz, 6))
        ax.plot(secs, mvs, c='black')

        ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
        ax.set_xlim(secs[0], secs[-1])
        ax.set_ylim(-1.5, 1.5)
        if use_idx:
            plt.xlabel('Samples')
        else:
            plt.xlabel('Time [s]')
        plt.ylabel('Voltage [mV]')
        plt.show()
