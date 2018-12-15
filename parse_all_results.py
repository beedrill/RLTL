from parseResult import parse_result
import os

folder = 'records-12408-simple'
files = os.listdir(folder)
waiting_time = [100 for i in range(0,24)]
waiting_time_equipped = [100 for i in range(0,24)]
waiting_time_unequipped = [100 for i in range(0,24)]

def get_smaller(l1, l2):
    return [min(e1,e2) for e1,e2 in zip(l1,l2)]


for file_name in files:
    ## ADD HOW TO DEAL WITH FILES HERE
    #file_name = "{}}/record_{}.txt".format(folder,net_itr)
    if file_name.startswith('record'):
        print(file_name)
        wl, wle, wlu = parse_result(folder+'/'+file_name)
        waiting_time = get_smaller(wl, waiting_time)
        waiting_time_equipped = get_smaller(wle, waiting_time_equipped)
        waiting_time_unequipped = get_smaller(wlu, waiting_time_unequipped)
print('------------------------------------------------')
print('------------------final results-----------------')
print(waiting_time)
print(waiting_time_equipped)
print(waiting_time_unequipped)
    ##
