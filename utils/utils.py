def counter_util(param, counter, val):
    '''
    counter helper
    
    params
    :param: list
        [start, end, interval]
    :counter: object
        counter records
    '''
    n = 1
    cur = param[0]
    while cur < param[1]:
        if val >= cur and val < cur+param[2]:
            counter[n] += 1
            break
        else:
            cur += param[2]
            n += 1
    if val == param[1]:
        counter[n-1] += 1