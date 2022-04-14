
def get_name(args: dict, **kwargs):
    """
        name the log file
        include the following info
        - algo
        - dataset
        - seed 
        - time_setting
        - num_epoch
        - clients_per_round
        - num_round
        - c0
        - num_clients
        - grad
        - model
        - name_type : 'log', 'grad'
    """
    filename = ""

    name_type = kwargs.pop('name_type', 'log')

    config = args.copy()
    ### IMPLEMENTATION ###
    # print(config)
    grad_name = "GRAD-%s-%s-E%s-K%s-N%s" % (
        config['dataset'],
        config['model'],
        config['num_epoch'],
        config['clients_per_round'],
        config['num_clients'],

    )

    if config['decay'] != 'round':
        grad_name += "-D%s" % config['decay']

    base_name = "%s-%s-SEED%s-TIME%s-E%s-K%s-R%s-C%s-N%s-Grad%s-%s" % (
        config['algo'],
        config['dataset'],
        config['seed'],
        config['time_seed'],
        config['num_epoch'],
        config['clients_per_round'],
        config['num_round'],
        config.pop('c0',0),
        config['num_clients'],
        config.pop('grad', ""),
        config['model'],
    )

    if name_type == 'log':
        filename = base_name + ".json"
    elif name_type == 'grad':
        filename = grad_name + ".json"
    else:
        raise Exception("Unknown name type")

    return filename