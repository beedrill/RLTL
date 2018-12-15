import os
penetration_rates = [0,0.2, 0.4,0.6,0.8,1]
names = ['00','02','04','06','08','10']
for pr, n in zip(penetration_rates, names):
    cmd = 'python run_multiagents_rltl.py --cpu --pysumo  --phase_representation original --simulator original --dir_name luxemboug-DUE-stationary-12408-pen{}   --penetration_rate {}'.format(n, pr)
    os.system(cmd)
