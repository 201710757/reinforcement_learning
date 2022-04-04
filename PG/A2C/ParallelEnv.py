import torch.multiprocessing as mp
import numpy as np
import gym


class ParallelEnv:
    def __init__(self, n_train_processes, env_name):
        self.env_name = env_name
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends
        
        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target = self.worker, args = (worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)
        
        for worker_end in worker_ends:
            worker_end.close()
        print("workerLen : ", len(self.workers))
    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]

        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
    
    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        #print(actions)
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]

        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

    def worker(self, worker_id, master_end, worker_end):
        master_end.close()
        env = gym.make(self.env_name)#'CartPole-v1')
        env.seed(worker_id)

        while True:
            cmd, data = worker_end.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                #ob = self.prepro(ob)
                worker_end.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                worker_end.send(ob)
            elif cmd == 'reset_task':
                ob = env.reset_task()
                worker_end.send(ob)
            elif cmd == 'close':
                worker_end.close()
                break
            elif cmd == 'get_spaces':
                worker_end.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError

    def prepro(I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I==144]=0
        I[I==109]=0
        I[I!=0]=1
        return I.astype(np.float).reshape(1,80,80)



