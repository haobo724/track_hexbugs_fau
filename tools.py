import re


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def str2int(v_str):
    temp = [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
    return temp


def sort_humanly(v_list):
    # temp = str2int(v_list)
    # print(temp)

    return sorted(v_list, key=str2int)

def delet_contours( contours, delete_list):
    # delta作用是offset，因为del是直接pop出去，修改长度了
    delta = 0
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    return contours