"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms
in the LICENSE file included with this software distribution.
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import json
import argparse


def json_dict(string):
    try:
        return json.loads(string)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string: {string}") from e


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "f", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


class BaseTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.hierachy = {}
        self.default_parameters = {}

    def parse(self, config_filename=''):
        self.initialize()

        _, _ = self.parser.parse_known_args()
        args = self.parser.parse_args()
        self.opt = args

        cmd_args = {}
        for key in self.default_parameters:
            if self.cmd2name(key) in args:
                cmd_args[self.cmd2name(key)] = getattr(self.opt, self.cmd2name(key))
            else:
                setattr(self.opt, self.cmd2name(key), self.default_parameters[key])

        if config_filename != '':
            self.load_from_json(config_filename)
        elif getattr(self.opt, 'config_filename') != '':
            config_fname = self.opt.config_filename
            self.load_from_json(config_fname)
            self.opt.config_filename = config_fname

        for key in cmd_args:
            setattr(self.opt, key, cmd_args[key])

        return self.opt

    @staticmethod
    def cmd2name(cmd):
        return cmd.replace('-', '_')

    @staticmethod
    def name2cmd(name):
        return name.replace('_', '-')

    def add_arg(self, cate, abbr, name, type, default):
        arg_type = str2bool if type is bool else type
        self.parser.add_argument('-' + abbr, '--' + self.name2cmd(name), type=arg_type, default=default)
        if cate not in self.hierachy.keys():
            self.hierachy[cate] = []
        self.hierachy[cate].append(name)

    def initialize(self):
        # base
        self.add_arg(cate='base', abbr='s', name='seed', type=str, default=0)
        self.add_arg(cate='base', abbr='g', name='gpu', type=str, default=0)

        # train
        self.add_arg(cate='train', abbr='b',   name='batch-size', type=int, default=2)
        self.add_arg(cate='train', abbr='lr',  name='learning-rate', type=float, default=1e-3)
        self.add_arg(cate='train', abbr='gmn', name='gradient-max-norm', type=float, default=-1.0)

        # data
        self.add_arg(cate='data', abbr='thread', name='thread-num', type=int, default=8)

        self.initialize_extra()
        self.initialized = True

    def initialize_extra(self):
        pass

    def save_json(self, save_path):
        data = {}
        for cate in self.hierachy.keys():
            data[cate] = {}
            for k in self.hierachy[cate]:
                data[cate][self.cmd2name(k)] = getattr(self.opt, self.cmd2name(k))
        with open(save_path, 'w') as fp:
            json.dump(data, fp, indent=4)
        print("saved options to json file: %s" % save_path)

    def load_from_json(self, json_path):
        with open(json_path) as fp:
            data = json.load(fp)
        for cate in data.keys():
            for k in data[cate].keys():
                setattr(self.opt, self.cmd2name(k), data[cate][self.cmd2name(k)])
        print("Options overwritten by json file: %s" % json_path)
