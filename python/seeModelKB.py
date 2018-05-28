# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six

import argparse
from cmd import Cmd
from ModelKB import Model


def unicode_text(s):
  if type(s) != six.text_type:
    return s.decode('utf-8')
  return s

class SeeModelCmd(Cmd):
  def __init__(self, model):
    Cmd.__init__(self)
    self.model = model
    self.k = 20

  def do_help(self, arg):
    print(" Command list:")
    print("\trole ROLE\tInspect role matrix")
    print("\tcalc EXPR\tCalculate vector")
    print("\tset  K   \tDisplay the top K results (default: 20)")
    print("\tquit     \tQuit")
    print()

  def do_role(self, role):
    try:
      self.model.show_m(unicode_text(role), self.k)
    except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print()

  def do_comprole(self, comprole):
    try:
      r1, r2 = unicode_text(comprole).split(' ')
      self.model.show_mm(r1, r2, self.k)
    except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print()

  def do_calc(self, expr):
    try:
      self.model.show_v(self.model.calc(unicode_text(expr)), self.k)
    except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print()

  def do_sim(self, s):
    try:
      x, y = unicode_text(s).split(' ~ ')
      print(self.model.calc(x).dot(self.model.calc(y)))
    except Exception as ex:
      template = "An exception of type {0} occurred. Arguments:\n{1!r}"
      message = template.format(type(ex).__name__, ex.args)
      print(message)
      print()

  def do_set(self, k):
    self.k = int(k)

  def do_quit(self, arg):
    raise SystemExit


def main():
  parser = argparse.ArgumentParser(description='See KB embedding model.')
  parser.add_argument('words_file', metavar='VOCAB_ENTITY', type=str,
                      help='counts of entities')
  parser.add_argument('roles_file', metavar='VOCAB_RELATION', type=str,
                      help='counts of relations')
  parser.add_argument('model_path', metavar='MODEL_PATH', type=str,
                      help='path to trained model')
  args = parser.parse_args()

  model = Model(args.words_file, args.roles_file, args.model_path)

  prompt = SeeModelCmd(model)
  prompt.prompt = '> '
  prompt.cmdloop()


if __name__ == '__main__':
  main()
