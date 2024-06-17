import numpy as np



alg = 'ac_lambda'


max_step = 5000
# max_step = 3000

lr = 0.0003
lamb = 0.8
beta = 0.01
is_minatar_origin = True

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
        cmd += ' --lamb=' + str(lamb)
        cmd += ' --beta=' + str(beta)

        cmd += ' --max-step=' + str(h)
        cmd += ' --gpu-no=' + str(cur_gpu_no)

        if is_minatar_origin:
            cmd += ' --is-minatar-origin'
        if is_save_data:
            cmd += ' --is-save-data'

        cmd += ' --wandb-offline'
        cmd += ' --seed=' + str(sd)

        # - log form
        cmd += ' > ./run_logs/ac/' + e + '_' + alg

        cmd += '_lr' + str(lr)
        cmd += '_lamb' + str(lamb)
        cmd += '_beta' + str(beta)

        cmd += '_imo' + str(int(is_minatar_origin))
        cmd += '_ms' + str(h)

        cmd += '_s' + str(sd) + '.log'
        cmd += ' 2>&1 &'

        print(cmd)
        cnt += 1

    print()


