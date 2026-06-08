import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from fedncf import *
from utils import *
from config import *
from data import SampleGenerator

config = get_config()
print(config)

if config['pretrain_path']:
    if config['use_individual_pretrain']:
        print(f"Loading individual pretrain from {config['pretrain_path']}")
    else:
        print(f"Loading shared pretrain from {config['pretrain_path']}")
else:
    print("Using random initialization")

seed_all(config['seed'])

trainer = FedTrainer(config)

initLogging()
logging.info(config)

rating = load_data(config)
sample_generator = SampleGenerator(config=config, ratings=rating)

test_recalls, test_ndcgs, test_hrs, final_test_round = trainer.run_experiment(config, sample_generator)

save_log_result(config, test_recalls, test_ndcgs, final_test_round)

best_hr = test_hrs[final_test_round]
logging.info(f'Best test HR: {best_hr} at round {final_test_round}')

if config['save_clients'] and not config['save_model']:
    logging.info(f"Saving client params to {config['clients_save_path']}")
    trainer.save_clients_params(config['clients_save_path'])

print("Experiment completed!")
