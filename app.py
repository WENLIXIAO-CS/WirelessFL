
## A bit of Setup
import argparse
from src.controllers.client import FedIoTClient 
from src.controllers.server import FedIoTServer

def fl_experiment(args, fedServer):
    ## config settings 
    fedServer.config_experiment(args)
    
    ## transfer config info to clients
    fedServer.init_clients()

    ## test speed of each client
    fedServer.test_comm_speed()

    fedServer.deploy_data()
    fedServer.train()
    # fedServer.wait()

def server_main():

    ##### <-- SET HYPERPARAMETER --> #####
    
    NUM_OF_CLIENTS = 40

    SEED_MAX = 7  # Num of Seed
    TSEED_MAX = 1 # Num of Time seed
    EXPERIMENT_FOLDER = 'test'

    ## Time Uniform ##

    args = {
        'clients_per_round'     : 4,
        'num_round'             : 1000,
        
        'num_epoch': 50,
        'batch_size': 24,
        'model': 'logistic',
        'update_rate': 0,
        'num_clients': NUM_OF_CLIENTS,
        'dataset': 'emnist_niid1_7_0_N1',
        'lr': 0.1,
        'wd': 0.001,
        'gpu': False,
        'noaverage': False,
        'experiment_folder': EXPERIMENT_FOLDER,
        'is_sys_heter': True,
        'eval_every': 5,
        'c0': 5,    ##
        'without_rp': False, ##
        'test_num':2,
        'decay' : 'round',
    }

    fedServer = FedIoTServer(args)
    fedServer.connect2clients()

    experimentConfig = {
        'seedMax': SEED_MAX,
    }
    for t_seed in range(1,TSEED_MAX+1):
        args['time_seed'] = t_seed
        for seed in range(1, 1+SEED_MAX):
            args['seed'] = seed


            args['algo'] = 'weighted'
            fedServer.start_experiment()
            fl_experiment(args, fedServer)

            args['algo'] = 'uniform'
            fedServer.start_experiment()
            fl_experiment(args, fedServer)

            # args['algo'] = 'proposed'
            # fedServer.start_experiment()
            # fl_experiment(args, fedServer)

            # args['algo'] = 'statistical'
            # fedServer.start_experiment()
            # fl_experiment(args, fedServer)

    fedServer.end_experiment()

    print("Experiment Ended.")

def client_main():
    fedClient = FedIoTClient()
    fedClient.connect2server()
    while fedClient.is_experiment_ongoing():
        fedClient.init_config()
        fedClient.config_experiment()
        fedClient.test_comm_speed()
        fedClient.deploy_data()
        fedClient.train()
    
    print("Experiment Ended.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='server or client')
    args = parser.parse_args()

    if args.mode == 'server':
        from src.controllers.server import FedIoTServer
        server_main()
    elif args.mode == 'client':
        client_main()
    else:
        raise Exception("Wrong parser parameter!")
