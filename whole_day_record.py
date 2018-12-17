import os
###use this to get hole day record for 24 hours
# for time in range(0,24):
#     os.system('python run_multiagents_rltl.py --mode test --load DQN_weights/luxemboug-DUE-12408-pen02-run16/DQN_SUMO_420000_weights --day_time {} --whole_day --penetration_rate 0.2 --cpu --record --sumo'
#     .format(time))

###use this to get whole day record for different penetration
names = ['00', '02', '04', '06', '08', '10']
penetration_rates = [0,0.2,0.4,0.6,0.8,1]
time = 8

for n,pr in zip(names,penetration_rates):
    os.system('python3 run_multiagents_rltl.py --cpu --sumo --mode test --record --load DQN_weights/Luxembourg-time{}-pen{}/DQN_SUMO_best_weights --penetration_rate {}'
    .format(time,n,pr))
