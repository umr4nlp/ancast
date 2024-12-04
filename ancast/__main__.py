#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Main entry point for script usage"""

import argparse
import logging
import random

from ancast import params
from ancast.evaluate import evaluate

logger = logging.getLogger(__name__)


def str2bool(v):
    # https://stackoverflow.com/a/43357954
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser("ancast++ Argparser")

    # paths
    parser.add_argument(
        '-p', '--pred', required=True, nargs='+',
        help='(required) path to prediction file or dir '
             '(if dir, same filenames must exist in `gold`)'
    )
    parser.add_argument(
        '-pe', '--pred-ext', nargs='+', default=['txt', 'amr', 'umr', 'pred'],
        help="if `args.pred` flag is a dir, which file extensions to consider"
    )
    parser.add_argument(
        '-g', '--gold', required=True, nargs='+',
        help='(required) path to gold file or dir '
             '(if dir, same filenames must exist in `pred`)'
    )
    parser.add_argument(
        '-ge', '--gold-ext', nargs='+', default=['txt', 'amr', 'umr', 'gold'],
        help="if `args.gold` flag is a dir, which file extensions to consider"
    )
    parser.add_argument(
        '-o',  '--output',
        help='(optional) path to output analysis files'
             '(if new dir, should not contain comma in the base dirname)',
    )

    # core
    # noinspection PyTypeChecker
    parser.add_argument(
        '-s', '--scope', type=str.lower, choices=['snt', 'doc'],
        help="whether to run ancast over AMR/UMR sentence-level graphs (`snt`) "
             "or to run ancast++ over UMR document-level graphs (`doc`)"
    )
    # noinspection PyTypeChecker
    parser.add_argument(
        '-df', '--data-format', type=str.lower, choices=['amr', 'umr'],
        help='whether the input data are AMR(s) or UMR(s) '
             '(automatically set to `umr` if `--scope` is `doc`)'
    )

    # hyperparams
    parser.add_argument(
        '-c', '--cneighbor', type=int,
        default=params.CNEIGHBOR,
    )
    parser.add_argument(
        '-sc', '--sense-coefficient', type=float,
        default=params.SENSE_COEFFICIENT,
    )
    parser.add_argument(
        '--separate-1-and-2', type=str2bool, nargs='?', const=True,
        default=params.SEPARATE_1_AND_2,
    )
    parser.add_argument(
        '--allow-reify', type=str2bool, nargs='?', const=True,
        default=params.ALLOW_REIFY,
    )
    parser.add_argument(
        '--use-alignment', type=str2bool, nargs='?', const=True,
        default=params.USE_ALIGNMENT,
    )
    parser.add_argument(
        '--use-smatch-top', type=str2bool, nargs='?', const=True,
        default=params.USE_SMATCH_TOP,
    )
    ### doc-level eval specific
    parser.add_argument(
        '--weighted', action='store_true',
        help="whether to apply weighted average for ancast++ doc-level evaluation"
    )

    # experiment
    parser.add_argument(
        '--seed', type=int, default=13681,
    )
    parser.add_argument(
        '--debug', action='store_true',
    )

    args = parser.parse_args()

    # script setup
    logging.basicConfig(
        format="[%(asctime)s][%(name)s][%(levelname)s]%(message)s",
        level=logging.DEBUG if args.debug else logging.INFO
    )
    random.seed(args.seed)

    ### inner
    return evaluate(
        scope=args.scope,
        pred_fpaths=args.pred,
        gold_fpaths=args.gold,
        output_fpath=args.output,
        pred_exts=args.pred_ext,
        gold_exts=args.gold_ext,
        data_format=args.data_format,
        Cneighbor=args.cneighbor,
        sense_coefficient=args.sense_coefficient,
        separate_1_and_2=args.separate_1_and_2,
        allow_reify=args.allow_reify,
        use_alignment=args.use_alignment,
        use_smatch_top=args.use_smatch_top,
        weighted_average=args.weighted,
    )


if __name__ == "__main__":
    main()