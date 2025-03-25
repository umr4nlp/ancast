#! /usr/bin/python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, List, Tuple, Union

import csv
import numpy as np

from ancast import io_utils, params
from ancast.document import DocumentMatch, SentenceMatch
from ancast.resource_utils import load_abt_vocab, load_reify_rels

logger = logging.getLogger(__name__)


def collect_fpaths_from_fdir(fpath: str, exts=None) -> List[str]:
    """depth of 1 at most,,,"""
    if os.path.isfile(fpath):
        return [fpath]

    out = []

    exts = exts or []
    exts_is_empty = len(exts) == 0
    if exts_is_empty:
        logger.warning("Received no filename extensions, collecting all files..")

    fdir = fpath
    for fname in sorted(os.listdir(fdir)):
        fpath = os.path.join(fdir, fname)
        ext = os.path.splitext(fpath)[1][1:]
        if os.path.isfile(fpath) and (ext in exts or exts_is_empty):
            out.append(fpath)
    return out

# noinspection PyShadowingBuiltins
def resolve_inputs(
        inputs,
        exts, delimiter="\n\n",
        return_fnames=False
) -> Union[List[str], Tuple[List[str], List[str]]]:
    if isinstance(inputs, str):
        inputs = [inputs]

    # load strings from fpaths
    fnames = []  # without extension
    inputs_resolved = []
    for input in inputs:
        if os.path.exists(input):
            # is a fpath, should be loaded
            pred_input_fpaths_list = collect_fpaths_from_fdir(input, exts=exts)

            for pred_input_fpath in pred_input_fpaths_list:
                inputs_resolved.append(
                    io_utils.load_txt(pred_input_fpath, delimiter=delimiter)
                )
                fnames.append(os.path.splitext(os.path.basename(pred_input_fpath))[0])
        else:
            # otherwise, should be a string input ready for eval
            inputs_resolved.append([input])
            fnames.append(None)

    if return_fnames:
        return inputs_resolved, fnames
    return inputs_resolved


# noinspection PyPep8Naming
def evaluate_snt(
        pred_inputs: Union[str, List[str]],
        gold_inputs: Union[str, List[str]],
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
        output_csv_as_string=False,
        **unused
) -> List[float]:
    # sentinel
    data_format = data_format or "umr"
    assert data_format in ['amr', 'umr'], \
        f"expected `amr` or `umr` data format, but received `{data_format}`"
    delimiter = "\n\n" if data_format == "amr" else "\n\n\n"

    # maybe load external resources
    if allow_reify:
        logger.debug("Loading external resources..")
        load_abt_vocab(global_cache=True)
        load_reify_rels(global_cache=True)

    pred_inputs_resolved, pred_fnames = resolve_inputs(
        pred_inputs, exts=pred_exts, delimiter=delimiter, return_fnames=True
    )
    gold_inputs_resolved = resolve_inputs(
        gold_inputs, exts=gold_exts, delimiter=delimiter
    )
    num_pred_inputs = len(pred_inputs_resolved)
    num_gold_inputs = len(gold_inputs_resolved)

    # there is no guarantee that the number of pred and gold inputs match perfectly
    if num_pred_inputs != num_gold_inputs:
        logger.warning(
            "Number of pred and gold inputs do not match (%d vs %d)",
            num_pred_inputs, num_gold_inputs
        )

    output_flag = output_fpath is not None

    # just iterate through pred inputs
    fscores = []
    for i, (pred_input_blocks, pred_fname) in enumerate(
            zip(pred_inputs_resolved, pred_fnames)):
        logger.debug("Current Pred File: `%s`", pred_fname)

        if i >= num_gold_inputs:
            logger.warning(
                "Found no matching gold input for `%s`, skipping..",
                pred_fname
            )
            continue

        csv_f = csv_writer = None
        if output_flag:
            write_title = True

            output_is_dir = os.path.isdir(output_fpath)
            if output_is_dir:
                pred_output = os.path.join(output_fpath, f'{pred_fname}.csv')
                csv_f = open(pred_output, 'w', newline='')
            else:
                pred_output = output_fpath
                if io_utils.csv_is_empty(pred_output):
                    write_title = False
                csv_f = open(pred_output, 'a', newline='')

            csv_writer = csv.writer(csv_f)
            if write_title:
                csv_writer.writerow(
                    [
                        "Sentence", data_format + "_test", data_format + "_gold",
                        "Match T2G", "Match G2T",
                        "Concept Match Precision", "Concept Match Recall", "Concept Match F Score",
                        "Labeled Relational Match Precision", "Labeled Relational Match Recall",
                        "LRM F-Score", "ULRM F-Score", "WLRM F-score", "Smatch Score"
                    ]
                )

        match = SentenceMatch(
            format=data_format,
            Cneighbor=Cneighbor,
            sense_coefficient=sense_coefficient,
            allow_reify=allow_reify,
            separate_1_and_2=separate_1_and_2,
            use_alignment=use_alignment,
            use_smatch_top=use_smatch_top,
        )

        # compute fscores while accumulating messages for analysis
        fscore = match.compute_scores(
            pred_inputs=pred_input_blocks,
            gold_inputs=gold_inputs_resolved[i]
        )
        fscores.append(fscore)

        if output_flag:
            for msg in match.msgs:
                csv_writer.writerow(msg)
            csv_f.close()
    
    if output_csv_as_string:
        return fscores, match.msgs
    else:
        return fscores

def evaluate_doc(
        pred_inputs: Union[str, List[str]],
        gold_inputs: Union[str, List[str]],
        output_fpath=None,
        pred_exts=None,
        gold_exts=None,
        Cneighbor: int = params.CNEIGHBOR,
        sense_coefficient: float = params.SENSE_COEFFICIENT,
        separate_1_and_2: bool = params.SEPARATE_1_AND_2,
        allow_reify: bool = params.ALLOW_REIFY,
        use_alignment: bool = params.USE_ALIGNMENT,
        use_smatch_top: bool = params.USE_SMATCH_TOP,
        weighted=None,
        output_csv_as_string=False,
        **unused
) -> Dict[str, Union[float, list[float]]]:
    # maybe load external resources
    if allow_reify:
        logger.debug("Loading external resources..")
        load_abt_vocab(global_cache=True)
        load_reify_rels(global_cache=True)

    pred_inputs_resolved, pred_fnames = resolve_inputs(
        pred_inputs, exts=pred_exts, delimiter="\n\n\n", return_fnames=True
    )
    gold_inputs_resolved = resolve_inputs(
        gold_inputs, exts=gold_exts, delimiter="\n\n\n"
    )
    num_pred_inputs = len(pred_inputs_resolved)
    num_gold_inputs = len(gold_inputs_resolved)

    # there is no guarantee that the number of pred and gold inputs match perfectly
    if num_pred_inputs != num_gold_inputs:
        logger.warning(
            "Number of pred and gold inputs do not match (%d vs %d)",
            num_pred_inputs, num_gold_inputs
        )

    output_flag = output_fpath is not None

    aggregate_flag = num_pred_inputs > 1
    aggregate = {
        'sent': [],
        'modal': [],
        'temporal': [],
        'coref': [],
        'comp': []
    }

    num_snts_by_docs, num_toks_by_docs = [], []
    for i, (pred_input_blocks, pred_fname) in enumerate(
            zip(pred_inputs_resolved, pred_fnames)):
        logger.debug("Current Pred File: `%s`", pred_fname)

        if i >= num_gold_inputs:
            logger.warning(
                "Found no matching gold input for `%s`, skipping..",
                pred_fname
            )
            continue

        csv_f = csv_writer = None
        if output_flag:
            write_title = True

            output_is_dir = os.path.isdir(output_fpath)
            if output_is_dir:
                pred_output = os.path.join(output_fpath, f'{pred_fname}.csv')
                csv_f = open(pred_output, 'w', newline='')
            else:
                pred_output = output_fpath
                if io_utils.csv_is_empty(pred_output):
                    write_title = False
                csv_f = open(pred_output, 'a', newline='')

            csv_writer = csv.writer(csv_f)
            if write_title:
                csv_writer.writerow(
                    [
                        "Sentence", "umr_test", "umr_gold",
                        "Match T2G", "Match G2T",
                        "Concept Match Precision", "Concept Match Recall", "Concept Match F Score",
                        "Labeled Relational Match Precision", "Labeled Relational Match Recall",
                        "LRM F-Score", "ULRM F-Score", "WLRM F-score", "Smatch Score"
                    ]
                )

        match = DocumentMatch(
            Cneighbor=Cneighbor,
            sense_coefficient=sense_coefficient,
            allow_reify=allow_reify,
            separate_1_and_2=separate_1_and_2,
            use_alignment=use_alignment,
            use_smatch_top=use_smatch_top,
        )

        # compute fscores while accumulating messages for analysis
        match.compute_scores(
            pred_inputs=pred_input_blocks,
            gold_inputs=gold_inputs_resolved[i]
        )
        for k,v in aggregate.items():
            v.append(getattr(match, f'{k}_fscore'))

        num_snts_by_docs.append(match.doc_num_snts)
        num_toks_by_docs.append(match.doc_num_toks)

        if output_flag:
            for msg in match.msgs:
                csv_writer.writerow(msg)
            csv_f.close()

    if aggregate_flag:
        logger.info("=== Aggregate Statistics ===")

        apply_weighted_average = weighted is not None
        for k,v in aggregate.items():
            weights = None
            if apply_weighted_average:
                weights = num_snts_by_docs if weighted == 'snts' else num_toks_by_docs

            aggregate[k] = cur_aggr = np.average(v, weights=weights)
            logger.info(f"{'Weighted ' if weighted else ''}Aggregate {k.capitalize()} Score:\t{cur_aggr:.2%}")

    else:
        for k, v in aggregate.items():
            aggregate[k] = v[0]
    if output_csv_as_string:
        return aggregate, match.msgs
    else:
        return aggregate

# @timer_decorator
def evaluate(*args, scope: str = "doc", **kwargs):
    # sentinel
    scope = scope.lower()
    assert scope in ['snt', 'doc'], \
        f'either `snt` or `doc` scope expected but received `{scope}`'

    eval_fn = evaluate_doc if scope == 'doc' else evaluate_snt
    return eval_fn(*args, **kwargs)
