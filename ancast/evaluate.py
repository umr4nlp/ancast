#! /usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import List

import numpy as np

from ancast import params
from ancast.document import DocumentMatch, SentenceMatch
from ancast.resource_utils import load_abt_vocab, load_reify_rels

logger = logging.getLogger(__name__)


def collect_fpaths_aux(fpaths: List[str], exts=None) -> List[str]:
    out = []
    exts = exts or []
    for fpath in fpaths:
        if os.path.isfile(fpath):
            out.append(fpath)
        else:
            if len(exts) == 0:
                logger.warning(
                    "Received a dir (`%s`) but no filename extensions, skipping..",
                    fpath
                )
                continue

            fdir = fpath
            for fname in os.listdir(fdir):
                fpath = os.path.join(fdir, fname)
                ext = os.path.splitext(fpath)[1][1:]
                if os.path.isfile(fpath) and ext in exts:
                    out.append(fpath)
    return out

def collect_fpaths(
        pred_fpaths,
        gold_fpaths,
        output_fpath=None,
        pred_exts=None,
        gold_exts=None,
) -> List[List]:
    # sentinel
    assert all(os.path.exists(x) for x in pred_fpaths)
    assert all(os.path.exists(x) for x in gold_fpaths)

    # collect all pred and gold fpaths as specified
    pred_fpath_list = collect_fpaths_aux(
        pred_fpaths, exts=pred_exts or []
    )
    gold_fpath_list = collect_fpaths_aux(
        gold_fpaths, exts=gold_exts or []
    )

    # NOTE: preserve `pred_fpaths` list
    pg_fpaths = []
    num_preds, num_golds = len(pred_fpath_list), len(gold_fpath_list)
    if num_preds != num_golds:
        logger.debug("Number of Files mismatch: p(%d) vs g(%d)", num_preds, num_golds)

        # match equal filenames (without considering extension)

        gold_basenames = [
            os.path.splitext(os.path.basename(x))[0] for x in gold_fpath_list
        ]

        for pred_fpath in pred_fpath_list:
            pred_basename = os.path.splitext(os.path.basename(pred_fpath))[0]

            found = False
            for i, gold_basename in enumerate(gold_basenames):
                found = pred_basename == gold_basename
                if found:
                    pg_fpaths.append(
                        [[pred_fpath, gold_fpath_list[i]]]
                    )
                    break

            if not found:
                logger.warning(
                    "Found a pred file (`%s`) without a matching gold file, skipping..",
                    pred_fpath
                )

    else:
        pg_fpaths = [
            [[pf, gf]] for pf, gf in zip(pred_fpath_list, gold_fpath_list)
        ]

    # maybe add output fpath
    has_output_fpath = output_fpath is not None
    if has_output_fpath:
        output_is_file = len(os.path.splitext(output_fpath)[-1]) > 1
        if output_is_file:
            for pg_pair in pg_fpaths:
                pg_pair.append(output_fpath)
        else:
            os.makedirs(output_fpath, exist_ok=True)
            for pg_pair in pg_fpaths:
                basename = os.path.splitext(os.path.basename(pg_pair[0][0]))[0]
                pg_pair.append(os.path.join(output_fpath, f'{basename}.csv'))

    return pg_fpaths

def evaluate_snt(
        pred_fpaths,
        gold_fpaths,
        data_format: str,
        output_fpath=None,
        pred_exts=None,
        gold_exts=None,
        Cneighbor: int = params.CNEIGHBOR,
        sense_coefficient: float = params.SENSE_COEFFICIENT,
        separate_1_and_2: bool = params.SEPARATE_1_AND_2,
        allow_reify: bool = params.ALLOW_REIFY,
        use_alignment: bool = params.USE_ALIGNMENT,
        use_smatch_top: bool = params.USE_SMATCH_TOP,
        **unused
):
    # sentinel
    data_format = data_format.lower()
    assert data_format in ['amr', 'umr']

    eval_tuples = collect_fpaths(
        pred_fpaths=pred_fpaths,
        pred_exts=pred_exts,
        gold_fpaths=gold_fpaths,
        gold_exts=gold_exts,
        output_fpath=output_fpath,
    )

    # maybe load external resources
    if allow_reify:
        logger.debug("Loading external resources..")
        load_abt_vocab(global_cache=True)
        load_reify_rels(global_cache=True)

    for eval_tuple in eval_tuples:
        DM = SentenceMatch(
            format=data_format,
            Cneighbor=Cneighbor,
            sense_coefficient=sense_coefficient,
            allow_reify=allow_reify,
            separate_1_and_2=separate_1_and_2,
            use_alignment=use_alignment,
            use_smatch_top=use_smatch_top,
        )
        output_csv = eval_tuple[1] if len(eval_tuple) > 1 else None
        DM.read_document(eval_tuple[0], output_csv=output_csv)

def evaluate_doc(
        pred_fpaths,
        gold_fpaths,
        output_fpath=None,
        pred_exts=None,
        gold_exts=None,
        Cneighbor: int = params.CNEIGHBOR,
        sense_coefficient: float = params.SENSE_COEFFICIENT,
        separate_1_and_2: bool = params.SEPARATE_1_AND_2,
        allow_reify: bool = params.ALLOW_REIFY,
        use_alignment: bool = params.USE_ALIGNMENT,
        use_smatch_top: bool = params.USE_SMATCH_TOP,
        weighted_average=False,
        **unused
):
    eval_tuples = collect_fpaths(
        pred_fpaths=pred_fpaths,
        pred_exts=pred_exts,
        gold_fpaths=gold_fpaths,
        gold_exts=gold_exts,
        output_fpath=output_fpath,
    )

    # maybe load external resources
    if allow_reify:
        logger.debug("Loading external resources..")
        load_abt_vocab(global_cache=True)
        load_reify_rels(global_cache=True)

    aggregate_flag = len(eval_tuples) > 1
    aggregate = {
        'sent': [],
        'modal': [],
        'temporal': [],
        'coref': [],
        'comp': []
    }

    valid_doc_nums = []
    for eval_tuple in eval_tuples:
        DM = DocumentMatch(
            Cneighbor=Cneighbor,
            sense_coefficient=sense_coefficient,
            allow_reify=allow_reify,
            separate_1_and_2=separate_1_and_2,
            use_alignment=use_alignment,
            use_smatch_top=use_smatch_top,
        )
        DM.read_document(eval_tuple[0], output_csv=eval_tuple[1])

        if aggregate_flag:
            for k,v in aggregate.items():
                v.append(getattr(DM, f'{k}_fscore'))
            valid_doc_nums.append(DM.doc_num_valid)

    if aggregate_flag:
        logger.info("=== Aggregate Statistics ===")
        for k,v in aggregate.items():
            cur_aggr = np.average(v, weights=valid_doc_nums if weighted_average else None)
            logger.info(f"{'Weighted ' if weighted_average else ''}Aggregate {k.capitalize()} Score:\t{cur_aggr:.2%}")

# @timer_decorator
def evaluate(scope: str, *args, **kwargs):
    # sentinel
    scope = scope.lower()
    assert scope in ['snt', 'doc'], \
        f'either `snt` or `doc` scope expected but received `{scope}`'

    eval_fn = evaluate_doc if scope == 'doc' else evaluate_snt
    eval_fn(*args, **kwargs)
