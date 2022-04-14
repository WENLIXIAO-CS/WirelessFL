###########################################
##  Benchmark : Uniform Sampling
###########################################

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch

criterion = torch.nn.CrossEntropyLoss()

class FedAvg9Trainer(BaseTrainer):

    def __init__(self, options, trainerConfig):
        model = choose_model(options)
        super(FedAvg9Trainer, self).__init__(options, trainerConfig)
        

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
            self.logExperimentInfo['gk'].append([c.Gk for c in self.clients])
            self.logExperimentInfo['acc'].append(self.acc)
            self.logExperimentInfo['global_loss'].append(self.log_round_loss)
            self.logExperimentInfo['round_time'].append(self.log_round_time)

            selected_clients = self.select_clients(seed=round_i)

            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)


            self.log_round_time = self.get_real_time(stats)

            # Aggregate
            self.latest_model = self.aggregate(solns)
            
            if self.decay == 'round':
                self.worker.optimizer.inverse_prop_decay_learning_rate(round_i)
            elif self.decay == 'soft':
                self.worker.optmizer.soft_decay_learning_rate()
            else:
                raise Exception("Wrong DECAY Method!")

            self.logExperimentInfo['sel_clients'].append([c.cid for c in selected_clients])

            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        # self.test_latest_model_on_evaldata(self.num_round)
        self.save_log()
        self.end_train()

    def compute_grad(self):  

        for c in self.clients:
            c_grad = self.get_grad(c)
            c.Gk = c_grad
    
        return

    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        for num_sample, local_solution in solns:
            averaged_solution += local_solution
        averaged_solution /= self.clients_per_round

        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

