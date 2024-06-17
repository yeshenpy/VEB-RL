import numpy as np



# alg = 'dqn'
# alg = 'double_dqn'
# alg = 'dsarsa'
alg = 'dqn_per'


# max_step = 5000
max_step = 3000

lr = 0.0003
alpha = 0.5
init_beta = 0.5
end_beta = 0.5
beta_anneal_horizon = 2000000
hr_interval = 1000

is_save_data = False

envs = [
    'breakout',
    'freeway',
    'asterix',
    'seaquest',
    'space_invaders',
]



seed_num = 3
gpu_num = 3
cnt = np.random.choice(range(gpu_num))

print('- Build commands for ' + alg + ' below:')
for e in envs:
    program = 'run_' + alg + '.py'

    if e == 'seaquest':
        h = 5000
    else:
        h = max_step

    for sd in range(seed_num):
        cur_gpu_no = cnt % gpu_num
        # cur_gpu_no = cnt % gpu_num + 2
        cmd = 'nohup python ' + program
        cmd += ' --env=' + e
        cmd += ' --lr=' + str(lr)

        cmd += ' --alpha=' + str(alpha)
        cmd += ' --init-beta=' + str(init_beta)
        cmd += ' --end-beta=' + str(end_beta)
        cmd += ' --bah=' + str(beta_anneal_horizon)

        cmd += ' --hard-replacement-interval=' + str(hr_interval)

        cmd += ' --max-step=' + str(h)
        cmd += ' --gpu-no=' + str(cur_gpu_no)

        if is_save_data:
            cmd += ' --is-save-data'

        cmd += ' --wandb-offline'
        cmd += ' --seed=' + str(sd)

        # - log form
        cmd += ' > ./run_logs/dqn/' + e + '_' + alg

        cmd += '_lr' + str(lr)
        cmd += '_alpha' + str(alpha)
        cmd += '_ibeta' + str(init_beta)
        cmd += '_ebeta' + str(end_beta)
        cmd += '_bah' + str(beta_anneal_horizon)
        cmd += '_hri' + str(hr_interval)

        cmd += '_ms' + str(h)

        cmd += '_s' + str(sd) + '.log'
        cmd += ' 2>&1 &'

        print(cmd)
        print('sleep 2')
        cnt += 1

    print()


