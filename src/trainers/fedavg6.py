###########################################
##  Proposed Sampling Scheme
###########################################

from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import numpy as np
import torch


criterion = torch.nn.CrossEntropyLoss()


class FedAvg6Trainer(BaseTrainer):
    def __init__(self, options, trainerConfig):
        model = choose_model(options)
        # self.move_model_to_gpu(model, options)

        # self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        # self.num_epoch = options['num_epoch']
        # worker = LrdWorker(model, self.optimizer, options)
        super(FedAvg6Trainer, self).__init__(options, trainerConfig)


    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()

        ## Compute Prob once before training
        self.prob = self.compute_grad_prob()

        for round_i in range(self.num_round):

            # Test latest model on train data
            # Test on Server
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Add update log 
            self.logExperimentInfo['qk'].append(self.prob.tolist())
            self.logExperimentInfo['gk'].append([c.Gk for c in self.clients])
            self.logExperimentInfo['acc'].append(self.acc)
            self.logExperimentInfo['global_loss'].append(self.log_round_loss)
            self.logExperimentInfo['round_time'].append(self.log_round_time)

            
            # Choose K clients (with replacement)
            if self.without_rp:
                print("Select without rp")
                selected_clients, repeated_times = self.select_clients_with_prob_without_replacement(seed=round_i)
            else:
                print("Select with rp")
                selected_clients, repeated_times = self.select_clients_with_prob(seed=round_i)
            
            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            self.log_round_time = self.get_real_time(stats)

            # Update latest model
            self.latest_model = self.aggregate(solns, repeated_times=repeated_times, clients=selected_clients)
            # self.optimizer.inverse_prop_decay_learning_rate(round_i)

            if self.decay == 'round':
                self.worker.optimizer.inverse_prop_decay_learning_rate(round_i)
            elif self.decay == 'soft':
                self.worker.optmizer.soft_decay_learning_rate()
            else:
                raise Exception("Wrong DECAY Method!")

            self.logExperimentInfo['sel_clients'].append([c.cid for c in selected_clients])


            # self.prob = self.compute_grad_prob()
   
        # Test final model on train data
        # self.test_latest_model_on_traindata(self.num_round)
        self.save_log()
        self.end_train()
        # self.test_latest_model_on_evaldata(self.num_round)

    def compute_grad_prob(self):  
        """Compute sampling prob by proposed scheme
            
            STEPS:
                1. Load gradient file (prepared)
                2. Used Matlab engine to compute prob
            
            return numpy.array

        """

        ## TODO : Extract self.grad_clients in base trainer

        assert self.grad_clients is not None 

        for i, c in enumerate(self.clients):
            c.Gk = self.grad_clients[i]

        gks = np.array([c.Gk for c in self.clients])
        pks = np.array([c.pk for c in self.clients])

        if self.is_sys_heter:
            print("System Heterogenity")
            A = np.array(self.client_times)
            B = gks**2 * pks**2
            probs = self.get_qk_matlab(A,B)
        else:
            print("Statistic heterogenity")
            raise NotImplementedError()

        for i, c in enumerate(self.clients):
            c.qk = probs[i]

        return probs

           
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
        sub = 0
        # averaged_solution = np.zeros(self.latest_model.shape)
        
        w0 = self.latest_model

        repeated_times = kwargs['repeated_times']
        clients = kwargs['clients']
        assert len(solns) == len(repeated_times)
        for i, (num_sample, local_solution) in enumerate(solns):
            c = clients[i]
            averaged_solution += local_solution * repeated_times[i] * c.pk / c.qk
            sub += repeated_times[i] * c.pk / c.qk

        averaged_solution /= self.clients_per_round
        sub /= self.clients_per_round
        
        averaged_solution = averaged_solution + w0*(1 - sub)

        return averaged_solution.detach()
