batch_size = 96
no_lr_steps = 2
lr_patience = 10
max_epoch = 100
update_epoch = 20

# best parameters
learning_rate = {'imagenet': 1E-2,
                'random': 1E-2,
                'imagenet2': 1E-2,
                'random2': 1E-2}

weight_decay = {'imagenet': 1E-7,
                'random': 1E-7,
                'imagenet2': 1E-7,
                'random2': 1E-7}

# hyperparameter optimization parameters
opt_interval = {'imagenet': [(1E-2, 1E0, 'log-uniform'), # learning rate
                            (1E-7, 1E-5, 'log-uniform')], # weight decay
                'random': [(1E-2, 1E0, 'log-uniform'), # learning rate
                            (1E-7, 1E-5, 'log-uniform')], # weight decay
                'imagenet2': [(1E-2, 1E0, 'log-uniform'), # learning rate
                            (1E-7, 1E-5, 'log-uniform')], # weight decay
                'random2': [(1E-2, 1E0, 'log-uniform'), # learning rate
                            (1E-7, 1E-5, 'log-uniform')],} # weight decay
                
no_random_starts = 3 # 8
no_opt_iters = 6 # 15

n_gpus = 4

if n_gpus == 2:
    os.environ['CUDA_VISIBLE_DEVICES']='0,2'