#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""resources"""

import importlib.resources
import json

ABT_VOCAB: dict = {}
REIFY_RELS: dict = {}
ALLOWED_TAGS: set = {"explicit"}

# resources
def load_resource(fname: str):
    return json.loads(
        importlib.resources.read_text('ancast.resources', fname)
    )

def load_abt_vocab(fname='abstract_vocab.json', global_cache=True):
    abt_vocab = load_resource(fname)
    if global_cache:
        global ABT_VOCAB
        ABT_VOCAB = abt_vocab
    return abt_vocab

def load_reify_rels(fname='reification_relations.json', global_cache=True):
    reify_rels = load_resource(fname)
    if global_cache:
        global REIFY_RELS
        REIFY_RELS = reify_rels
        ALLOWED_TAGS.add('reified')
    return reify_rels
