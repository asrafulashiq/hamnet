import argparse
import shutil
import os


def parse_args(parser=None):
    if parser is not None:
        parser = argparse.ArgumentParser(parents=[parser],
                                         conflict_handler='resolve')
    else:
        parser = argparse.ArgumentParser(conflict_handler='resolve')

    # ----------------------------------- path ----------------------------------- #
    parser.add_argument('--log_path', type=str, default='./logs')

    # ----------------------------------- model ---------------------------------- #
    parser.add_argument("--model_name",
                        default="base",
                        help="name to save model")
    parser.add_argument("--feature_type", type=str, default="I3D")
    parser.add_argument("--num_class", type=int, default=20)
    parser.add_argument("--feature_fps", type=int, default=25)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--feature_size", type=int, default=2048)
    parser.add_argument("--num_segments", type=int, default=750)
    parser.add_argument('--ckpt',
                        type=str,
                        default=None,
                        help='the path of pre-trained model file')
    parser.add_argument("--suffix", type=str, default="")
    # ---------------------------------- dataset --------------------------------- #
    parser.add_argument("--dataset_name", type=str, default="Thumos14reduced")

    # -------------------------------- hyperparams ------------------------------- #
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--class_thresh', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5,
                        help='learning rates for steps(list form)')
    parser.add_argument("--max_epochs", type=int, default=100)

    # ----------------------------------- other ---------------------------------- #
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed (-1 for no manual seed)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--unsave_latest', action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--gpus", type=str, default='0', help="gpu-id")

    parser.add_argument("--disable_logfile",
                        action="store_true",
                        help="whether to log to a file")
    parser.add_argument("--progress_refresh", type=int, default=20)

    return parser
