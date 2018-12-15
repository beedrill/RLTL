def parse_result(filename = "record.txt"):
    f = open(filename, 'r')
    #f = open("record-DUA-pen02.txt", 'r')
    line  = f.readline()
    waiting_time_list = [0 for i in range(0,24)]
    waiting_time_list_equipped = [0 for i in range(0,24)]
    waiting_time_list_unequipped = [0 for i in range(0,24)]
    n_v = [0 for i in range(0,24)]
    n_v_equipped = [0 for i in range(0,24)]
    n_v_unequipped = [0 for i in range(0,24)]
    def get_hour(depart_time):
        if depart_time<1824:
            index = 0
        elif depart_time>88223:
            index = 23
        else:
            index = int((depart_time-1824)/3600)
        return index
    while line:
        equipped = None
        temp = line.split()
        vid = temp[0]
        depart_time = float(temp[1])
        waiting_time = float(temp[2])
        if len(temp)>3:
            equipped = temp[3]
        hour = get_hour(depart_time)
        if equipped == 'True':
            n_v_equipped[hour] += 1
            waiting_time_list_equipped[hour]+= waiting_time
        elif equipped == 'False':
            n_v_unequipped[hour]+=1
            waiting_time_list_unequipped[hour]+= waiting_time

        n_v[hour] +=1
        waiting_time_list[hour]+= waiting_time

        line = f.readline()
    f.close()
    try:
        wl = None
        wle = None
        wlu = None
        print(n_v)

        wl = [waiting_time_list[i]/n_v[i] if n_v[i]>0 else 0 for i in range(0,24)]
        print( wl)
        print('equipped waiting time')
        wle = [waiting_time_list_equipped[i]/n_v_equipped[i] if n_v[i]>0 else 0 for i in range(0,24)]
        print( wle)
        print('unequipped waiting time')
        wlu = [waiting_time_list_unequipped[i]/n_v_unequipped[i] if n_v[i]>0 else 0 for i in range(0,24)]
        print( wlu)
    except:
        wl = wl or [0 for i in range(0,24)]
        wle = wle or [0 for i in range(0,24)]
        wlu = wlu or [0 for i in range(0,24)]
        print("exception")
    return wl, wle, wlu
if __name__ == '__main__':
    parse_result()
