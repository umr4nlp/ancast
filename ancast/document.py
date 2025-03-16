#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Document representation"""

import csv
import logging
import re
from collections import defaultdict

from ancast import params
from ancast.closure import get_cluster
from ancast.matches import MatchResolution, Match
from ancast.metric import Metric
from ancast.ops import protected_divide, parse_alignment
from ancast.sentence import Sentence

logger = logging.getLogger(__name__)

SUBPATTERN = r'\(((\S*)\s*(\:\S*)\s*(\S*?))\)'


def get_edge_list_temp(tuple_dict):
    edge_list = defaultdict(list)
    # this will sometimes be problematic if debugging is calling "graph", this defaultdict
    for key, value in tuple_dict.items():
        if value == "after":
            edge_list[key[0]].append((key[1], "r"))
        elif value == "before":
            edge_list[key[1]].append((key[0], "r"))
        elif value == "depends-on":
            edge_list[key[0]].append((key[1], "dp"))
        elif value in {"contained", "includes"}:
            edge_list[key[1]].append((key[0], "dn"))
            edge_list[key[0]].append((key[1], "up"))
        elif value == "overlap":   # overlap sorted by number size
            min_key = min(key[0], key[1])
            max_key = max(key[0], key[1])
            edge_list[min_key].append((max_key, "o"))
        else:
            logger.info(f"{value} is an unrecognized relation in temporal!")

    return edge_list

def get_edge_list_coref(tuple_dict):
    edge_list = defaultdict(list)
    for key, value in tuple_dict.items():
        if value == "same-entity":
            edge_list[key[0]].append((key[1], "sn"))
            edge_list[key[1]].append((key[0], "sn"))
        elif value == "same-event":
            edge_list[key[1]].append((key[0], "sv"))
            edge_list[key[1]].append((key[0], "sv"))
        elif value == "subset-of":
            edge_list[key[0]].append((key[1], "up"))
            edge_list[key[1]].append((key[0], "dn"))
        else:
            logger.info("Unrecognized relation in coref!")

    return edge_list

def add_var(var_name, my_dict, other_dict, translate, match_list, n):
    if var_name not in my_dict:
        other = translate(match_list, var_name)
        if var_name in match_list.keys():
            my_dict[var_name] = n
            my_dict[n] = var_name
            if other != "NULL":
                other_dict[other] = n
                other_dict[n] = other
        else:
            # var_name is not a var, but const like author, root ...
            if var_name != other:
                logger.info(f"Translation error: inconsistant placeholder name {var_name} and {other} between test and gold!")
            my_dict[var_name] = n
            my_dict[n] = var_name
            other_dict[other] = n
            other_dict[n] = other
        n += 1

    return n


class SentenceMatch:

    # noinspection PyPep8Naming
    def __init__(
            self,
            format: str = 'umr',
            Cneighbor: int = params.CNEIGHBOR,
            sense_coefficient: float = params.SENSE_COEFFICIENT,
            allow_reify: bool = params.ALLOW_REIFY,
            separate_1_and_2: bool = params.SEPARATE_1_AND_2,
            use_alignment: bool = params.USE_ALIGNMENT,
            use_smatch_top: bool = params.USE_SMATCH_TOP,
    ):
        self.format = format
        self.Cneighbor = Cneighbor
        self.sense_coefficient = sense_coefficient
        self.allow_reify = allow_reify
        self.separate_1_and_2 = separate_1_and_2
        self.use_alignment = use_alignment
        self.use_smatch_top = use_smatch_top

        self.sentences = []
        self.smatch_scores = []
        self.concept_scores = []
        self.labeled_scores = []
        self.unlabeled_scores = []
        self.weighted_labeled_scores = []

        self.semantic_metric_precision = Metric("precision")
        self.semantic_metric_recall = Metric("recall")

        self.var_n = 0

        ### number of text data
        self.num_snts = 0
        self.num_toks = 0

        # final score
        self.sent_fscore = -1

        # per-anno messages
        self.msgs = []

    def macro_avg(self, M):
        self.smatch_scores.append(M.smatch_format_score)
        self.concept_scores.append(M.concept_match_fscore)
        self.labeled_scores.append(M.lbd_fscore)
        self.unlabeled_scores.append(M.ulbd_fscore)
        self.weighted_labeled_scores.append(M.wlbd_fscore)

    def add_doct_info(self, M, **kwargs):
        self.semantic_metric_precision.log_and_inc_metric(M.Mt01)
        self.semantic_metric_recall.log_and_inc_metric(M.Mt10)

    # def output_to_csv(self, writer, M = None, title=False):
    def summarize(self, match_res):
        formatted_str_match01 = '\n'.join(f"{Match.gname(match_res.umr0, key)}: {Match.gname(match_res.umr1, value)} \t Q-level = {match_res.quality_list01[key]}" for key, value in match_res.match_list01.items()) + "\n\nGood quality percentage:\t{:.2%}".format(match_res.gp01)
        formatted_str_match10 = '\n'.join(f"{Match.gname(match_res.umr1, key)}: {Match.gname(match_res.umr0, value)} \t Q-level = {match_res.quality_list10[key]}" for key, value in match_res.match_list10.items()) + "\n\nGood quality percentage:\t" + "{:.2%}".format(match_res.gp10)
        self.msgs.append(
            [
                match_res.umr0.text,
                match_res.umr0.generate_umr_text(),
                match_res.umr1.generate_umr_text(),
                formatted_str_match01,
                formatted_str_match10,
                "{:.2%}".format(match_res.match_score01),
                "{:.2%}".format(match_res.match_score10),
                "{:.2%}".format(match_res.concept_match_fscore),
                "{:.2%}".format(match_res.lbd_p),
                "{:.2%}".format(match_res.lbd_r),
                "{:.2%}".format(match_res.lbd_fscore),
                "{:.2%}".format(match_res.ulbd_fscore),
                "{:.2%}".format(match_res.wlbd_fscore),
                "{:.2%}".format(match_res.smatch_format_score)
            ]
        )

    def compute_scores(self, pred_inputs, gold_inputs):
        name, bgi = 0, -1
        for pred_input in pred_inputs:
            bgi += 1
            if "#" not in pred_input:
                logger.warning(f"Encountered unknown block in %s, skipping", bgi)
                continue

            pred_input_match = re.search(self.snt_pat, pred_input)
            if pred_input_match:  # the block is crucial
                while not re.search(self.snt_pat, gold_inputs[bgi]):
                    bgi += 1
                gold_input = gold_inputs[bgi]
                gold_input_match = re.search(self.snt_pat, gold_input)

                name += 1
                snt_test = pred_input.split(pred_input_match.group())[1].split("\n")[0].strip()
                snt_gold = gold_input.split(gold_input_match.group())[1].split("\n")[0].strip()
                # if snt_test != snt_gold:
                #     logger.info(f"{name} sentence is not the same!")

                if self.format_is_amr:
                    amr_test = "\n".join(pred_input.split("#")[-1].split("\n")[1:])
                    amr_gold = "\n".join(gold_input.split("#")[-1].split("\n")[1:])
                else:
                    amr_test = "\n".join(pred_input.split("\n\n")[1].split("\n")[1:])
                    amr_gold = "\n".join(gold_input.split("\n\n")[1].split("\n")[1:])
                ta, ga = {}, {}

                tamr = Sentence(
                    sent=snt_test,
                    semantic_text=amr_test,
                    alignment=ta,
                    sent_num=name,
                    format=self.format
                )
                gamr = Sentence(
                    sent=snt_gold,
                    semantic_text=amr_gold,
                    alignment=ga,
                    sent_num=name,
                    format=self.format
                )

                if tamr.invalid or gamr.invalid:
                    logger.warning(f"Error encountered, skipping sentence {name}")
                    continue

                match_res = MatchResolution(
                    tamr,
                    gamr,
                    Cneighbor=self.Cneighbor,
                    sense_coefficient=self.sense_coefficient,
                    separate_1_and_2=self.separate_1_and_2,
                    use_alignment=self.use_alignment,
                    use_smatch_top=self.use_smatch_top,
                )

                self.macro_avg(match_res)
                self.add_doct_info(match_res)  # micro-average

                # cache msgs for analysis
                self.summarize(match_res)

        # aggregate
        ps, rs = self.semantic_metric_precision.compute("lr"), self.semantic_metric_recall.compute("lr")
        self.sent_fscore = protected_divide(2*ps*rs, ps+rs)
        logger.info(
            f"Sent Micro:\tPrecision: {ps:.2%}\tRecall: {rs:.2%}\tFscore: {self.sent_fscore:.2%}"
        )

        return self.sent_fscore

    @property
    def format_is_amr(self):
        return  self.format == "amr"

    @property
    def snt_pat(self):
        return  r"::snt\s" if self.format_is_amr else r"::\ssnt\d+"


class DocumentMatch(SentenceMatch):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.doc_annotations_test = {"temporal":  {}, "modal": {}, "coref": {}}
        self.doc_annotations_gold = {"temporal":  {}, "modal": {}, "coref": {}}
        self.doc_var_list = {"test" : {}, "gold":{}}

        self.doc_num_snts = -1
        self.doc_num_toks = -1

        # final scores
        self.modal_fscore = -1
        self.temporal_fscore = -1
        self.coref_fscore = -1
        self.comp_fscore = -1

        # per-anno messages
        self.msgs = []

    def add_doct_info(self, M, **kwargs):
        super().add_doct_info(M, **kwargs)

        test_doc = kwargs["test_doc"]
        gold_doc = kwargs["gold_doc"]

        patterns = {
            'temporal': r':temporal\s*\(((\(.*\:.*\)\s*)*)\)',
            'modal': r':modal\s*\(((\(.*\:.*\)\s*)*)\)',
            'coref': r':coref\s*\(((\(.*\:.*\)\s*)*)\)'
        }

        for key, pattern in patterns.items():
                match = re.search(pattern, test_doc)
                if match:
                    ms = match.group(1)
                    rs = re.findall(SUBPATTERN, ms)
                    for mts in rs:
                        head = mts[1]
                        rel = mts[2]
                        tail = mts[3]
                        self.var_n = add_var(head, self.doc_var_list["test"], self.doc_var_list["gold"], M.translate_match, M.match_list01, self.var_n)
                        self.var_n = add_var(tail, self.doc_var_list["test"], self.doc_var_list["gold"], M.translate_match, M.match_list01, self.var_n)
                        self.doc_annotations_test[key][(self.doc_var_list["test"][head], self.doc_var_list["test"][tail])] = rel[1:]

                match = re.search(pattern, gold_doc)
                if match:
                    ms = match.group(1)
                    rs = re.findall(SUBPATTERN, ms)
                    for mts in rs:
                        head = mts[1]
                        rel = mts[2]
                        tail = mts[3]
                        self.var_n = add_var(head, self.doc_var_list["gold"], self.doc_var_list["test"], M.translate_match, M.match_list10, self.var_n)
                        self.var_n = add_var(tail, self.doc_var_list["gold"], self.doc_var_list["test"], M.translate_match, M.match_list10, self.var_n)
                        self.doc_annotations_gold[key][(self.doc_var_list["gold"][head], self.doc_var_list["gold"][tail])] = rel[1:]

    def compute_scores(self, pred_inputs, gold_inputs):
        name = num_snts = num_toks = 0

        # ignoring file head
        for pred_input, gold_input in zip(pred_inputs, gold_inputs):
            name += 1
            logger.debug("Sentence %d"%name)

            try:
                assert "sentence" in pred_input and "sentence" in gold_input, \
                    f"Keyword `sentence` is not found in block {name}"
            except AssertionError as error:
                logger.info(f"Format Error: {error.args[0]}")
                raise
            try:
                assert "document" in pred_input and "document" in gold_input, \
                    f"Keyword `document` is not found in block {name}"
            except AssertionError as error:
                logger.info(f"Format Error: {error.args[0]}")
                raise

            t_buff = pred_input.split("# sentence level graph:")
            g_buff = gold_input.split("# sentence level graph:")

            t_sent = re.sub(r'^\d+[\s\t]*', '', t_buff[0]).strip()
            g_sent = re.sub(r'^\d+[\s\t]*', '', g_buff[0]).strip()

            t_buff = t_buff[1].strip().split("# alignment:")
            g_buff = g_buff[1].strip().split("# alignment:")

            t_graph = t_buff[0].strip()
            g_graph = g_buff[0].strip()

            t_buff = t_buff[1].strip().split("# document level annotation:")
            g_buff = g_buff[1].strip().split("# document level annotation:")

            t_alignment = t_buff[0].strip()
            g_alignment = g_buff[0].strip()

            t_doclevel = t_buff[1].strip()
            g_doclevel = g_buff[1].strip()

            t_alignment = parse_alignment(t_alignment)
            g_alignment = parse_alignment(g_alignment)

            if t_sent != g_sent:
                logger.info(f"Warning: Sentence {name} sentence content mismatched! Using test sentence.")
                g_sent = t_sent

            tumr = Sentence(
                sent = t_sent,
                semantic_text = t_graph,
                alignment= t_alignment,
                sent_num = name,
                format = self.format
            )
            gumr = Sentence(
                sent = g_sent,
                semantic_text = g_graph,
                alignment= g_alignment,
                sent_num = name,
                format = self.format
            )

            if tumr.invalid or gumr.invalid:
                logger.info(f"Error encountered, skipping sentence {name}")
                continue

            match_res = MatchResolution(
                tumr,
                gumr,
                Cneighbor=self.Cneighbor,
                sense_coefficient=self.sense_coefficient,
                separate_1_and_2=self.separate_1_and_2,
                use_alignment=self.use_alignment,
                use_smatch_top=self.use_smatch_top,
            )

            self.add_doct_info(match_res, test_doc = t_doclevel, gold_doc = g_doclevel)
            self.macro_avg(match_res)

            self.summarize(match_res)

            num_snts += 1
            if t_sent is not None:
                num_toks += len(t_sent.split())

        self.doc_num_snts = num_snts
        self.doc_num_toks = num_toks

        ps, rs = self.semantic_metric_precision.compute("lr"), self.semantic_metric_recall.compute("lr")
        self.sent_fscore = protected_divide(2*ps*rs, ps+rs)
        logger.info(f"Sent Micro:\tPrecision: {ps:.2%}\tRecall: {rs:.2%}\tFscore: {self.sent_fscore:.2%}")

        pm, rm, self.modal_fscore = self.calculate_modal()
        pt, rt, self.temporal_fscore = self.calculate_TCTC("temporal", get_edge_list_temp)
        pc, rc, self.coref_fscore = self.calculate_TCTC("coref", get_edge_list_coref)

        logger.info(f"Modality:\tPrecision: {pm:.2%}\tRecall: {rm:.2%}\tFscore: {self.modal_fscore:.2%}")
        logger.info(f"Temporal:\tPrecision: {pt:.2%}\tRecall: {rt:.2%}\tFscore: {self.temporal_fscore:.2%}")
        logger.info(f"Coref:\t\tPrecision: {pc:.2%}\tRecall: {rc:.2%}\tFscore: {self.coref_fscore:.2%}")

        comp_p = self.semantic_metric_precision.compute("document")
        comp_r = self.semantic_metric_recall.compute("document")

        self.comp_fscore = protected_divide(2*comp_p*comp_r, comp_p+comp_r)

        logger.info(f"Comprehensive Score:\t{self.comp_fscore:.2%}\n")

        return self.sent_fscore, self.modal_fscore, self.temporal_fscore, self.coref_fscore, self.comp_fscore


    def warn_error(self, theset, test_or_gold, temp_or_coref):
        if theset:
            elem = next(iter(theset))
            if len(elem) == 1:
                logger.info(f"These nodes have erratic circular relations in {temp_or_coref} in {test_or_gold} file.")
                pstr = " ".join([self.doc_var_list[test_or_gold][n[0]] for n in theset])
            elif len(elem) == 2:
                logger.info(f"These pairs of nodes have paradoxical relations in {temp_or_coref} in {test_or_gold} file.")
                pstr = "\t".join(["("+self.doc_var_list[test_or_gold][n[0]]+", "+self.doc_var_list[test_or_gold][n[1]]+")" for n in theset])
            elif len(elem) == 3:
                logger.info(f"These nodes triples have contradictory coreferential relations in {test_or_gold}.")
                pstr = "\t".join(["(" + ", ".join([self.doc_var_list[test_or_gold][n[i]] for i in range(3)])  +")" for n in theset])

            logger.info(pstr)

    def calculate_modal(self):
        set0, set1 = set(), set()
        for key, value in self.doc_annotations_test["modal"].items():
            set0.add(str(key)+str(value))
        for key,value in self.doc_annotations_gold["modal"].items():
            set1.add(str(key)+str(value))

        set2 = set0 & set1
        modal_precision = protected_divide(len(set2) , len(set0))
        modal_recall    = protected_divide(len(set2) , len(set1))

        self.semantic_metric_precision.log_and_inc("modal", "score",len(set2), len(set0))
        self.semantic_metric_recall.log_and_inc("modal", "score", len(set2), len(set1))

        return modal_precision, modal_recall, protected_divide(2*modal_precision*modal_recall , (modal_precision + modal_recall))

    def calculate_TCTC(self, keyword, get_f):
        """
        1. Organize temporal relation all into uni-direction
        2. Make this uni-directed graph a edge list;
        3. Get this edge list into Closure and use DFS to get a closure
        4. Use the result of two closure to calculate

        depends-on, overlap
        contain (only going down), after, before
        """

        el_test =  get_f(self.doc_annotations_test[keyword])
        el_gold =  get_f(self.doc_annotations_gold[keyword])

        # two steps above are pure translation
        cluster_el_test, test_whole_set, inconsistent_test, circular_test, wrongcorefs_test = get_cluster(el_test)
        cluster_el_gold, gold_whole_set, inconsistent_gold, circular_gold, wrongcorefs_gold = get_cluster(el_gold)

        self.warn_error(inconsistent_test, "test", keyword)
        self.warn_error(inconsistent_gold, "gold", keyword)
        self.warn_error(circular_test, "test", keyword)
        self.warn_error(circular_gold, "gold", keyword)
        self.warn_error(wrongcorefs_test, "test", keyword)
        self.warn_error(wrongcorefs_gold, "gold", keyword)

        s0 = 0
        cnt_test = 0
        for cluster, associated_nodes in cluster_el_test:
            cluster_size = len(associated_nodes)
            s0 += cluster_size *  protected_divide(len(cluster & gold_whole_set) , len(cluster))
            # here we need to be careful that there may be some ill-formed standing-alone pairs
            cnt_test += cluster_size

        self.semantic_metric_precision.log_and_inc(keyword, "score", s0, cnt_test)
        s0 = protected_divide(s0, cnt_test)

        s1 = 0
        cnt_gold = 0
        for cluster, associated_nodes in cluster_el_gold:
            cluster_size = len(associated_nodes)
            s1 += cluster_size *  len(cluster & test_whole_set) / len(cluster)
            cnt_gold += cluster_size
        self.semantic_metric_recall.log_and_inc(keyword, "score", s1, cnt_gold)
        s1 = protected_divide(s1, cnt_gold)

        return s0, s1, protected_divide(2*s0*s1 , (s0+s1))
