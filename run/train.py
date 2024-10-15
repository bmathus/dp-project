from argparse import ArgumentParser

from project.experiment import Experiment

def main(cfg):
    # Start new training experiment
    experiment = Experiment(cfg)
    experiment.start_experiment()

if __name__ == "__main__":
    p = ArgumentParser()

    # Experiment
    p.add_argument("--name", "-n", type=str, default="basic", help="Experiment name")
    p.add_argument("--ver", "-v", type=str, default="", help="Experiment version (neptune custom id) for resuming run, leave empty for new run")
    p.add_argument("--data_path", "-d", type=str, default="./data/ACDC", help="Path to dataset")

    # p.add_argument("--checkpoint_freq","-cf", type=int, default=2, help="Frequency of saving model & opt checkpoints")
    p.add_argument('--seed', type=int,  default=1337, help='Random seed')
    p.add_argument('--deterministic',"-dt", type=int,  default=1,help='Whether use deterministic training')

    p.add_argument("--max_iter","-e", type=int, default=30000, help="Number of iterations")
    p.add_argument("--base_lr","-lr", type=float, default=0.01, help="Segmentation network learning rate")
    p.add_argument('--num_classes', type=int, default=4, help='Output channel of network')

    # Data & labels
    p.add_argument("--batch_size","-bs", type=int, default=24, help="Batch size per GPU")
    p.add_argument('--labeled_bs', type=int, default=12,help='Labeled batch_size per GPU')
    p.add_argument('--labeled_num', type=int, default=7, help='Labeled data')
    p.add_argument('--patch_size', type=int,  default=256,help='patch size of network input')

    # Costs
    p.add_argument('--consistency', type=float,default=0.1, help='consistency')
    p.add_argument('--consistency_rampup', type=float,default=200.0, help='consistency_rampup')


    # p.add_argument("--num_workers","-nw", type=int, default=2, help="Number of Dataloader workers")
    # p.add_argument("--learning_rate","-lr", type=float, default=0.1, help="Optimizer learning rate")
    # p.add_argument("--num_hidden","-nh", type=int, default=500, help="Number of hidden units")

    cfg = p.parse_args()
    main(cfg)