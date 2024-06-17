################################################################################################################
# Authors:                                                                                                     #
# Hongyao Tang (bluecontra@tju.edu.cn)                                                                            #
#                                                                                                              #
# python run_random_agent.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################

import os, time, argparse, random
import numpy as np
from minatar import Environment
import torch
from agents.dqn import DQN,Genome
import wandb
from agents.mod_neuro_evo import SSNE
from agents.ES import sepCEM
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_env, eval_episodes=10, env_name=None):
    avg_return = 0.
    avg_step = 0.
    for _ in range(eval_episodes):
        eval_env.reset()
        s = eval_env.state()
        done = False
        t = 0
        ep_reward = 0
        while not done:
            action = policy.select_action(s, is_greedy=True)
            reward, done = eval_env.act(action)
            s = eval_env.state()
            avg_return += reward
            avg_step += 1

            ep_reward += reward
            t += 1

            # BugFixed for Seaquest never done case
            
            
            if env_name is not None and env_name == 'seaquest':
                if t >= 1000 and ep_reward < 0.1:
                #if t >= 10000 :#and ep_reward < 0.1:
                    break

    avg_return /= eval_episodes
    avg_step /= eval_episodes

    print("Evaluation over %d episodes - Score: %f, Steps: %f" % (eval_episodes, avg_return, avg_step))
    return avg_return, avg_step


def get_args_from_parser():
    parser = argparse.ArgumentParser()

    # breakout, freeway, asterix, seaquest, space_invaders
    parser.add_argument("--env", type=str, default='seaquest', help='env')
    parser.add_argument("--seed", type=int, default=1, help='random_seed')
    parser.add_argument("--evo_iters", type=int, default=100, help='random_seed')
    parser.add_argument("--max-step", type=int, default=5000, help='total_steps(k)')
    parser.add_argument("--gpu-no", type=str, default='1', help='gpu_no')

    parser.add_argument("--rl_to_ea_synch_period", type=int, default=1, help='rl_to_ea_synch_period')


    parser.add_argument("--lr", type=float, default=0.0003, help='learning_rate')
    parser.add_argument("--frac", type=float, default=0.1, help='learning_rate')
    parser.add_argument("--elite_size", type=float, default=0.2, help='elite_size')

    parser.add_argument("--ti", type=int, default=1, help='train_interval')
    parser.add_argument("--hard-replacement-interval", type=int, default=1000, help='hard replacement interval')
    parser.add_argument("--pop_size", type=int, default=5, help='pop_size')
    parser.add_argument("--EA_target_update_freq", type=int, default=1, help='EA_target_update_freq')
    parser.add_argument("--eval-interval", type=int, default=50000, help='number of steps per evaluation point')
    parser.add_argument("--is-save-data", action="store_true", help='is_save_data')
    parser.add_argument("--wandb-offline", action="store_true", help='set offline wandb mode')

    parser.add_argument('-sigma_init', default=1e-3, type=float)
    parser.add_argument('-damp', default=1e-3, type=float)
    parser.add_argument('-damp_limit', default=1e-5, type=float)
    parser.add_argument('-elitism', action='store_true')

    return parser.parse_args()


def rollout(args, env, agent , RL_agent ,Random, num_actions):
    env.reset()
    s = env.state()
    done = False

    ep_reward = 0
    ep_step_count = 0

    while (not done):
        # Interact with env
        if Random:
            a = np.random.choice([i for i in range(num_actions)])
        else:
            a = agent.select_action(s)
            agent.epsilon_decay()
        r, done = env.act(a)
        s_ = env.state()
        RL_agent.store_experience(s, a, r, s_, int(done))
        ep_step_count += 1
        s = s_
        ep_reward += r

        # BugFixed for Seaquest never done case
        if args.env == 'seaquest':
            if ep_step_count >= 1000 and ep_reward < 0.1:
            #if ep_step_count >= 10000 :#and ep_reward < 0.1:
                break
    return ep_reward, ep_step_count
import torch.nn.functional as F
def caculate_TD_error(Pop, RL_agent, device, gamma):
    batch_samples = RL_agent.replay_buffer.sample(int(1024*5))

    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.FloatTensor(np.stack(batch_samples.state, axis=0)).to(device).permute(0, 3, 1, 2).contiguous()
    actions = torch.LongTensor(np.stack(batch_samples.action, axis=0)).to(device).unsqueeze(1)
    rewards = torch.FloatTensor(np.stack(batch_samples.reward, axis=0)).to(device).unsqueeze(1)
    next_states = torch.FloatTensor(np.stack(batch_samples.next_state, axis=0)).to(device).permute(0, 3, 1, 2).contiguous()
    dones = torch.FloatTensor(np.stack(batch_samples.is_terminal, axis=0)).to(device).unsqueeze(1)

    error_list = []
    for agent in Pop:
        Q_sa = agent.Q_net(states).gather(1, actions)
        Q_sa_ = agent.Q_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute the target
        target = rewards + gamma * (1 - dones) * Q_sa_
        loss = F.mse_loss(target, Q_sa)
        error_list.append(-loss.data.cpu().numpy())

    return np.squeeze(error_list)

def rl_to_evo(rl_agent, evo_net):
    for target_param, param in zip(evo_net.Q_net.parameters(), rl_agent.Q_net.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(evo_net.Q_target.parameters(), rl_agent.Q_target.parameters()):
        target_param.data.copy_(param.data)

import copy
def run_exp(args):
    #####################  hyper parameters  ####################

    MAX_TOTAL_STEPS = 1000 * args.max_step
    INIT_RANDOM_STEPS = 10000

    # Init wandb logger
    # Default hyperparam (partially) from MinAtar Paper
    hyperparam_config = dict(
        alg_name='dqn',
        total_max_steps=MAX_TOTAL_STEPS,
        init_random_steps=INIT_RANDOM_STEPS,
        batch_size=32,
        memory_size=100000,
        epsilon_decay_steps=100000,
        init_epsilon=1.0,
        end_epsilon=0.1,
    )
    hyperparam_config.update(vars(args))
    # offline mode is safer and more reliable for wandb, yet need manual sync
    if args.wandb_offline:
        os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        project="Min_Atari",
        job_type=str(hyperparam_config['env']) + '_' + hyperparam_config['alg_name'],
        entity="tju-lpy",
        notes="common baseline",
        tags=["baseline", "dqn"],
        config=hyperparam_config,
    )
    pop_size = args.pop_size
    elite_size = args.elite_size
    wandb.run.name =  "CEM_VEB_pop_size_"+ str(pop_size) + "_"+ str(elite_size)+ "CEM_"+str(args.sigma_init) + "_"+ str(args.damp)+"_Update_EA_target_"+str(args.EA_target_update_freq)+"_Train_"+str(args.evo_iters)+"_" +"_Frac_"+str(args.frac) + "_"+ str(args.rl_to_ea_synch_period)+"_"+str(hyperparam_config['env']) + '_' + hyperparam_config['alg_name']
    wandb.run.save()



    # Init Env
    env = Environment(args.env, random_seed=args.seed)
    # Get channels and number of actions specific to each game
    state_shape = env.state_shape()
    num_actions = env.num_actions()

    env4eval = Environment(args.env, random_seed=args.seed * 1234)

    print('-- Env:', args.env)
    print('-- Seed:', args.seed)
    print('-- Configurations:', vars(args))

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    # tf.set_random_seed(seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_no
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Init RL agent
    Pop = []
    for i in range(pop_size):
        gen = Genome(state_shape=state_shape, num_actions=num_actions, device=device, learning_rate=args.lr,
                     hard_replacement_interval=args.hard_replacement_interval)  # #epsilon_decay_steps=int(100000/6.0))
        Pop.append(gen)

    Mu_agent = Genome(state_shape=state_shape, num_actions=num_actions, device=device, learning_rate=args.lr,
                hard_replacement_interval=args.hard_replacement_interval)  # #epsilon_decay_steps=int(100000/6.0))

    CEM = sepCEM(Pop[0].Q_net.get_size(), mu_init=Pop[0].Q_net.get_params(),
                      sigma_init=args.sigma_init, damp=args.damp,
                      damp_limit=args.damp_limit,
                      pop_size=pop_size, antithetic=not pop_size % 2, parents=pop_size // 2,
                      elitism=args.elitism)




    agent = DQN(state_shape=state_shape, num_actions=num_actions, device=device,
                learning_rate=args.lr, hard_replacement_interval=args.hard_replacement_interval, epsilon_decay_steps=int(100000/(elite_size*pop_size+1)))

    return_his, step_his = [], []
    q_loss_his, avg_q_loss_his = [], []
    evaluation_return_list, evaluation_step_list = [], []

    global_step_count = 0
    ep_num = 0

    # Initial evaluation
    print("---------------------------------------")
    print('- Evaluation at Step', global_step_count, '/', MAX_TOTAL_STEPS)
    eval_return, eval_step = evaluate_policy(agent, eval_env=env4eval, eval_episodes=10, env_name=args.env)
    evaluation_return_list.append(eval_return)
    evaluation_step_list.append(eval_step)
    wandb.log({'eval_return': eval_return}, step=global_step_count)
    wandb.log({'eval_step': eval_step}, step=global_step_count)
    print("---------------------------------------")

    EA_generation = 0
    previous_print_steps = 0

    # Training
    best_agents = list(range(int(pop_size*elite_size)))

    EA_avg_error = []
    EA_best_error = []
    EA_worst_error = []
    replace_index = -1
    while global_step_count < MAX_TOTAL_STEPS:

        generation_steps = 0
        fitness = np.zeros(pop_size)
        if global_step_count >= INIT_RANDOM_STEPS:
            EA_generation +=1
            for i in best_agents:
                ep_reward, ep_step_count = rollout(args,env,Pop[i],RL_agent=agent, Random=False, num_actions=num_actions)
                generation_steps +=ep_step_count
                fitness[i] +=ep_reward

        # RL inter
        ep_reward, ep_step_count = rollout(args,env,agent,RL_agent=agent, Random=global_step_count < INIT_RANDOM_STEPS, num_actions=num_actions)

        generation_steps += ep_step_count
        global_step_count +=generation_steps
        # RL train
        if global_step_count >= INIT_RANDOM_STEPS:
            for _ in range(min(global_step_count-INIT_RANDOM_STEPS, generation_steps)):
                loss = agent.train(Pop,Pop[np.argmax(fitness)].Q_net, np.max(fitness))
                q_loss_his.append(loss)

        es_params = CEM.ask(pop_size)
        if replace_index != -1:
            es_params[replace_index] = Pop[replace_index].Q_net.get_params()

        for i in range(pop_size):
            Pop[i].Q_net.set_params(es_params[i])

        if global_step_count >= INIT_RANDOM_STEPS:
            for _ in range(args.evo_iters):
                td_error_list = caculate_TD_error(Pop, agent,device,0.99)

                new_td_error_list = copy.deepcopy(td_error_list)
                new_td_error_list = -new_td_error_list
                new_td_error_list.sort()
                EA_avg_error.append(np.mean(new_td_error_list))
                EA_best_error.append(np.mean(new_td_error_list[:2]))
                EA_worst_error.append(np.mean(new_td_error_list[-2:]))
                sorted_id = sorted(range(len(td_error_list)), key=lambda k: td_error_list[k], reverse=True)
                CEM.tell(es_params, td_error_list)
                # print(td_error_list)
                # print("sorted_id",replace_index, sorted_id)
                best_agents = sorted_id[:int(elite_size*pop_size)]

        Mu_agent.Q_net.set_params(CEM.mu)

        if EA_generation% args.EA_target_update_freq == 0 :
            for ind in Pop:
                ind.Q_target.load_state_dict(ind.Q_net.state_dict())

        if global_step_count - previous_print_steps >= args.eval_interval:
            previous_print_steps = global_step_count
            print("---------------------------------------")
            print('- Evaluation at Step', global_step_count, '/', MAX_TOTAL_STEPS)
            eval_return, eval_step = evaluate_policy(agent, eval_env=env4eval, eval_episodes=10, env_name=args.env)
            evaluation_return_list.append(eval_return)
            evaluation_step_list.append(eval_step)
            wandb.log({'rl_eval_return': eval_return}, step=global_step_count)
            wandb.log({'rl_eval_step': eval_step}, step=global_step_count)


            ea_eval_return, ea_eval_step = evaluate_policy(Mu_agent, eval_env=env4eval, eval_episodes=10, env_name=args.env)
            wandb.log({'ea_eval_step': ea_eval_step}, step=global_step_count)
            wandb.log({'ea_eval_return': ea_eval_return}, step=global_step_count)

            eval_step_list = [ea_eval_step, eval_step]
            wandb.log({'eval_step': eval_step_list[np.argmax([ea_eval_return,eval_return])]}, step=global_step_count)
            wandb.log({'eval_return': np.max([ea_eval_return,eval_return])}, step=global_step_count)

            wandb.log({'EA_avg_error': np.mean(EA_avg_error)}, step=global_step_count)
            wandb.log({'EA_best_error': np.mean(EA_best_error)}, step=global_step_count)
            wandb.log({'EA_worst_error': np.mean(EA_worst_error)}, step=global_step_count)

            EA_avg_error = []
            EA_best_error = []
            EA_worst_error = []
            print("---------------------------------------")


        if ep_num % args.rl_to_ea_synch_period == 0 and EA_generation > 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(td_error_list)
            assert replace_index not in best_agents
            rl_to_evo(agent, Pop[replace_index])
            print('Sync from RL --> Nevo')
        else :
            replace_index = -1

        ep_num += 1
        return_his.append(ep_reward)
        step_his.append(ep_step_count)

        if ep_num % 10 == 0:
            avg_ep_return = sum(return_his[-10:]) / len(return_his[-10:])
            avg_ep_step = sum(step_his[-10:]) / len(step_his[-10:])
            avg_q_loss = sum(q_loss_his[-100:]) / len(q_loss_his[-100:]) if agent.update_cnt > 0 else 0
            avg_q_loss_his.append(avg_q_loss)
            print('- Steps:', global_step_count, '/', MAX_TOTAL_STEPS,
                  'Ep:', ep_num, ', return:', avg_ep_return, 'ep_steps:', avg_ep_step, 'avg_q_loss:', avg_q_loss)
            wandb.log({'training_return': avg_ep_return}, step=global_step_count)
            wandb.log({'training_step': avg_ep_step}, step=global_step_count)
            wandb.log({'q_loss': avg_q_loss}, step=global_step_count)

    if args.is_save_data:
        print('=========================')
        print('- Saving data.')

        save_folder_path = './results/dqn'
        save_folder_path += '_lr' + str(args.lr)
        save_folder_path += '_hrfreq' + str(args.hard_replacement_interval)
        save_folder_path += '_ti' + str(args.ti)
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        file_index = save_folder_path + args.env
        file_index += '_h' + str(args.max_step)
        file_index += '_s' + str(args.seed)
        np.savez_compressed(file_index,
                            q_loss=avg_q_loss_his,
                            eval_return=eval_return,
                            eval_step=eval_step,
                            config=vars(args),
                            )

        print('- Data saved.')
        print('-------------------------')


if __name__ == '__main__':

    t1 = time.time()
    arguments = get_args_from_parser()
    run_exp(args=arguments)
    print('Running time: ', time.time() - t1)
