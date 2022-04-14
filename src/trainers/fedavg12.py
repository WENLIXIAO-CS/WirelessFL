from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from src.optimizers.gd import GD
import torch


criterion = torch.nn.CrossEntropyLoss()

# full sample 

class FedAvg12Trainer(BaseTrainer):
    def __init__(self, options, trainerConfig):
        # model = choose_model(options)
        # self.move_model_to_gpu(model, options)

        # self.optimizer = GD(model.parameters(), lr=options['lr'], weight_decay=options['wd'])
        # self.num_epoch = options['num_epoch']
        # worker = LrdWorker(model, self.optimizer, options)
        super(FedAvg12Trainer, self).__init__(options, trainerConfig)
        
        self.clients_per_round = self.numOfClients
        print("clients per round", self.clients_per_round)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        
        # self.test_latest_model_on_traindata(round_i)

        for round_i in range(self.num_round):

            # Test latest model on train data
            self.test_latest_model_on_traindata(round_i)
            self.test_latest_model_on_evaldata(round_i)

            # Add update log 
            self.logExperimentInfo['global_loss'].append(self.log_round_loss)
            self.logExperimentInfo['acc'].append(self.acc)
            self.logExperimentInfo['round_time'].append(self.log_round_time)
            self.logExperimentInfo['sel_clients'].append([c.cid for c in selected_clients])


            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)


            # Solve minimization locally
            solns, stats = self.local_train(round_i, selected_clients)

            self.log_round_time = self.get_real_time(stats)
            # Track communication cost
            # self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)

            if self.decay == 'round':
                self.worker.optimizer.inverse_prop_decay_learning_rate(round_i)
            elif self.decay == 'soft':
                self.worker.optmizer.soft_decay_learning_rate()
            else:
                raise Exception("Wrong DECAY Method!")


            # if self.decay == 'inverse':
            #     self.optimizer.inverse_prop_decay_learning_rate(round_i)
            # elif self.decay == 'soft':
            #     self.optimizer.soft_decay_learning_rate()
            # else:
            #     raise Exception("Wrong Decay METHOD!")

        self.save_log()
        self.end_train()



    def aggregate(self, solns, **kwargs):
        averaged_solution = torch.zeros_like(self.latest_model)
        # averaged_solution = np.zeros(self.latest_model.shape)
        # assert self.simple_average
        num_sigma = 0
        for num_sample, local_solution in solns:
            averaged_solution += local_solution * num_sample
            num_sigma += num_sample
        # averaged_solution /= self.clients_per_round
        averaged_solution /= num_sigma
        # averaged_solution = from_numpy(averaged_solution, self.gpu)
        return averaged_solution.detach()

  