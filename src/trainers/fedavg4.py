###########################################
##  Benchmark : Weighted Sampling
###########################################

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch

criterion = torch.nn.CrossEntropyLoss()

class FedAvg4Trainer(BaseTrainer):
    def __init__(self, options, trainerConfig):

        model = choose_model(options)
        super(FedAvg4Trainer, self).__init__(options, trainerConfig)

        self.prob = self.compute_prob()


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)
            
            self.compute_grad()
            
            # Add update log 
            self.logExperimentInfo['qk'].append(self.prob.tolist())
            self.logExperimentInfo['gk'].append([c.Gk for c in self.clients])
            self.logExperimentInfo['acc'].append(self.acc)
            self.logExperimentInfo['global_loss'].append(self.log_round_loss)
            self.logExperimentInfo['round_time'].append(self.log_round_time)


            # Choose K clients prop to data size
            if self.simple_average:
                if self.without_rp:
                    print("Select without rp")
                    selected_clients, repeated_times = self.select_clients_with_prob_without_replacement(seed=round_i)
                else:
                    print("Select with rp")
                    selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)
            else:
                selected_clients = self.select_clients(seed=round_i)
                repeated_times = None

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            self.log_round_time = self.get_real_time(stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times)
            
            if self.decay == 'round':
                self.worker.optimizer.inverse_prop_decay_learning_rate(round_i)
            elif self.decay == 'soft':
                self.worker.optmizer.soft_decay_learning_rate()
            else:
                raise Exception("Wrong DECAY Method!") 
                
            self.logExperimentInfo['sel_clients'].append([c.cid for c in selected_clients])

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        # self.test_latest_model_on_evaldata(self.num_round)

        self.save_log()
        self.end_train()

    def compute_prob(self):
        probs = []
        for c in self.clients:
            probs.append(len(c.train_data))

        probs = np.array(probs)/sum(probs)

        return probs

    def compute_grad(self):  

        for c in self.clients:
            c_grad = self.get_grad(c)
            c.Gk = c_grad
    
        return

    def select_clients_with_prob(self, seed=1):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed)
        index = np.random.choice(len(self.clients), num_clients, p=self.prob)
        index = sorted(index.tolist())

        select_clients = []
        select_index = []
        repeated_times = []
        for i in index:
            if i not in select_index:
                select_clients.append(self.clients[i])
                select_index.append(i)
                repeated_times.append(1)
            else:
                repeated_times[-1] += 1
        return select_clients, repeated_times

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        if self.simple_average:
            repeated_times = kwargs['repeated_times']
            assert len(solns) == len(repeated_times)
            for i, (num_sample, local_solution) in enumerate(solns):
                averaged_solution += local_solution * repeated_times[i]
            averaged_solution /= self.clients_per_round
        else:
            for num_sample, local_solution in solns:
                averaged_solution += num_sample * local_solution
            averaged_solution /= self.all_train_data_num
            averaged_solution *= (self.numOfClients/self.clients_per_round)

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

