batch_size = 96
no_lr_steps = 2
lr_patience = 2 # 10
max_epoch = 10 # 100

# best parameters
learning_rate = {'imagenet': 0.0001,
                'random': 0.001}

weight_decay = {'imagenet': 0.0005,
                'random': 0.0005}

# hyperparameter optimization parameters
opt_interval = {'imagenet': [(1E-4, 1E-2, 'log-uniform'), # learning rate
                            (1E-6, 1E-4, 'log-uniform')], # weight decay}
                'random': [(1E-4, 1E-2, 'log-uniform'), # learning rate
                          (1E-6, 1E-4, 'log-uniform')]} # weight decay
                
no_random_starts = 3 # 8
no_opt_iters = 6 # 15
