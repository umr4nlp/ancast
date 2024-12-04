#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Various operations"""

import numpy as np
from ancast.params import SENSE_COEFFICIENT


def string_sim(str1a, str2a):
    str1 = str1a.lower()
    str2 = str2a.lower()

    if (str1 not in str2) and (str2 not in str1):
        return 0.0
    elif str1 in str2:
        return len(str1) / len(str2)
    elif str2 in str1:
        return len(str2) / len(str1)
    else:
        raise NotImplementedError

def jaccard_sim(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_index = len(intersection) / len(union) if union else 0
    return jaccard_index

def name_sim(str1, str2, sense_sim, sense_coefficient=SENSE_COEFFICIENT):
    return string_sim(str1, str2) * (1 + sense_coefficient * (sense_sim - 1))

def rel_weight(a, b):
    return np.sqrt(a*b) + 1

def get_argcnt(edge_matrix, node):
    node_set = set()
    q = [node]
    while q:
        c = q.pop(0)
        nexts = np.where(edge_matrix[c] > 0)
        for i in nexts[0].tolist():
            if (i not in node_set):
                node_set.add(i)
                q.append(i)
    return len(node_set)

def get_children(edge_matrix, from_node, to_node):
    # traverse, one node counted twice
    from_node_set = set()
    to_node_set = set()
    q = [from_node]
    while q:
        c = q.pop(0)
        nexts = np.where(edge_matrix[c] > 0)
        for i in nexts:
            if (i not in from_node_set) and (i != to_node):
                from_node_set.add(i)
                q.append(i)
        q = [from_node]
    q = [to_node]
    while q:
        c = q.pop(0)
        nexts = np.where(edge_matrix[c] > 0)
        for i in nexts:
            if (i not in to_node_set):
                to_node_set.add(i)
                q.append(i)
    return len(from_node_set), len(to_node_set)

def protected_divide(a, b):
    if b==0:
        return 0.0
    else:
        return a / b

def parse_alignment(alignment_text):
    ras = {}
    if not alignment_text:
        return ras
    als = alignment_text.strip().split("\n")
    for l in als:
        if "alignment" in l:
            continue
        v, al = l.strip().split(":")
        ras[v] = al.strip()
    return ras

# def rel_hash(word1, rel, word2):
#     return word1.var+"-"+rel+"-"+word2.var, word2.var+"-"+reverse(rel)+"-"+word1.var

# ======================== relation ===========================
def reverse(rel):
    if rel[-3:] == "-of":
        return rel[:-3]
    else:
        return rel + "-of"

def is_reverse(rel):
    if rel == "mod":        # mod is "domain-of"
        return True
    elif rel == "mod-of":
        return False
    elif rel[-3:] == "-of":
        return True
    else:
        return False

# ================= for label match =================
def rel_label_set_comp(rel_label_set, target_label_set):
    """
    For possible fine-grained match.
    """
    set_a = rel_label_set
    set_b = target_label_set
    return len(set_a & set_b) # simply count fully equal labels
