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

def collect_fpaths(
        pred_fpaths,
        gold_fpaths,
        output_fpath=None,
        pred_exts=None,
        gold_exts=None,
) -> List[List]:
    if isinstance(pred_fpaths, str):
        pred_fpaths = [pred_fpaths]
    if isinstance(gold_fpaths, str):
        gold_fpaths = [gold_fpaths]

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

    return fscores

def evaluate_doc(
        pred_fpaths: Union[str, List[str]],
        gold_fpaths: Union[str, List[str]],
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
        **unused
) -> Dict[str, Union[float, list[float]]]:
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

    num_snts_by_docs = []
    num_toks_by_docs = []
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

        for k,v in aggregate.items():
            v.append(getattr(DM, f'{k}_fscore'))

        num_snts_by_docs.append(DM.doc_num_snts)
        num_toks_by_docs.append(DM.doc_num_toks)

    if aggregate_flag:
        logger.info("=== Aggregate Statistics ===")

        apply_weighted_average = weighted is not None
        for k,v in aggregate.items():
            weights = None
            if apply_weighted_average:
                weights = num_snts_by_docs if weighted == 'snts' else num_toks_by_docs

            aggregate[k] = cur_aggr = np.average(v, weights=weights)

            logger.info(f"{'Weighted ' if weighted else ''}Aggregate {k.capitalize()} Score:\t{cur_aggr:.2%}")

    elif len(eval_tuples) == 1:
        for k, v in aggregate.items():
            aggregate[k] = v[0]

    return aggregate

# @timer_decorator
def evaluate(*args, scope: str = "doc", **kwargs):
    # sentinel
    scope = scope.lower()
    assert scope in ['snt', 'doc'], \
        f'either `snt` or `doc` scope expected but received `{scope}`'

    eval_fn = evaluate_doc if scope == 'doc' else evaluate_snt
    return eval_fn(*args, **kwargs)


if __name__ == '__main__':
    pred_amr = """# ::id 0
# ::annotator bart-amr
# ::snt Resolutely support the thread starter! I compose a poem in reply:
(z0 / multi-sentence
    :snt1 (z1 / support-01
              :mode imperative
              :ARG0 (z2 / you)
              :ARG1 (z3 / person
                        :ARG0-of (z4 / start-01
                                     :ARG1 (z5 / thread)))
              :manner (z6 / resolute))
    :snt2 (z7 / compose-02
              :ARG0 (z8 / i)
              :ARG1 (z9 / poem)
              :ARG2-of (z10 / reply-01
                            :ARG0 z8)))"""
    gold_amr = """# ::id bolt12_64556_5627.1 ::date 2012-12-04T17:55:20 ::annotator SDL-AMR-09 ::preferred
# ::snt Resolutely support the thread starter! I compose a poem in reply:
# ::save-date Sun Dec 8, 2013 ::file bolt12_64556_5627_1.txt
(m / multi-sentence
      :snt1 (s / support-01 :mode imperative
            :ARG0 (y / you)
            :ARG1 (p / person
                  :ARG0-of (s2 / start-01
                        :ARG1 (t / thread)))
            :manner (r / resolute))
      :snt2 (r2 / reply-01
            :ARG0 (i / i)
            :ARG2 (c / compose-02
                  :ARG0 i
                  :ARG1 (p2 / poem))))"""

    print(pred_amr)
    print(gold_amr)

    fscores = evaluate(
        # pred_amr,
        # gold_amr,
        "samples/umr_test.txt",
        "samples/umr_gold.txt",
        data_format="umr",
        scope="snt"
    )
    print(fscores)
