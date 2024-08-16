from argparse import ArgumentParser

from project.experiment import Experiment

def main(cfg):
    # Start new training experiment
    experiment = Experiment(cfg)
    experiment.train()

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--name", "-n", type=str, default="basic", help="Experiment name")
    p.add_argument("--ver", "-v", type=str, default="run.1", help="Experiment version")
    p.add_argument("--checkpoint_freq","-cf", type=int, default=5, help="Frequency of saving model & opt checkpoints")

    p.add_argument("--batch_size","-bs", type=int, default=16, help="Batch size")
    p.add_argument("--num_workers","-nw", type=int, default=2, help="Number of Dataloader workers")

    p.add_argument("--learning_rate","-lr", type=float, default=0.1, help="Optimizer learning rate")
    p.add_argument("--max_epochs","-e", type=int, default=10, help="Number of epochs to train")
    p.add_argument("--num_hidden","-nh", type=int, default=128, help="Number of hidden units")

    cfg = p.parse_args()
    main(cfg)