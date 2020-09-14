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
    parser.add_argument('--output-path', type=str, default='./outputs')
    parser.add_argument('--log-path', type=str, default='./logs')

    # ----------------------------------- model ---------------------------------- #
    parser.add_argument("--model-name",
                        default="base",
                        help="name to save model")
    parser.add_argument("--feature-type", type=str, default="I3D")
    parser.add_argument("--num-class", type=int, default=20)
    parser.add_argument("--feature-fps", type=int, default=25)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--feature-size", type=int, default=2048)
    parser.add_argument("--num-segments", type=int, default=750)
    parser.add_argument('--modal',
                        type=str,
                        default='all',
                        choices=['rgb', 'flow', 'all'])
    parser.add_argument('--ckpt',
                        '--model_file',
                        type=str,
                        default=None,
                        help='the path of pre-trained model file')
    parser.add_argument("--suffix", type=str, default="")
    # ---------------------------------- dataset --------------------------------- #
    # parser.add_argument("--dataset-name", type=str, default="Thumos14reduced")
    parser.add_argument("--dataset-name", type=str, default="Thumos14reduced")

    # -------------------------------- hyperparams ------------------------------- #
    parser.add_argument('--alpha', type=float, default=1e-4)
    parser.add_argument('--class-thresh', type=float, default=0.15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='learning rates for steps(list form)')
    parser.add_argument('--num-iters', type=int, default=10000)
    parser.add_argument("--max-epoch", type=int, default=200)

    # ----------------------------------- other ---------------------------------- #
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed (-1 for no manual seed)')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--unsave-latest', action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--gpus", type=str, default='0')
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_separate", action="store_true")
    parser.add_argument("--plot_total_images", type=int, default=60)

    parser.add_argument("--disable-logfile",
                        action="store_true",
                        help="whether to log to a file")
    parser.add_argument("--progress-refresh", type=int, default=20)
    parser.add_argument("--cas_with_attn", type=str, default="true")

    return parser
