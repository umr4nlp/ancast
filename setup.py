#! /usr/bin/python3
# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name='ancast',
    version='0.1.0',
    description='AnCast++ metric for evaluating UMR semantic graphs.',
    # url='https://github.com/sxndqc/ancast',
    url='https://github.com/umr4nlp/ancast',
    # include_package_data=True,
    # package_data={'resources': ['*.json']},
    packages=setuptools.find_packages(exclude=['refs']),
    install_requires=[
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
    ],
)
