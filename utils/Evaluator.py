# coding: utf-8

import sys

dataDir = '../../VQA'

from VQAeval import VQAEval
import json
import random
import yaml
import os
from VQAhelper import VQAhelper


class Evaluator:

    def __init__(self, annFile, quesFile, ids_to_labels_file, results, already_decoded=False, res_file='decoded_results.json'):
        # annFile and quesFile are predefined for each subset of the dataset (train and val)
        # annFile     ='../data/miniCOCO_val_annotations.json'
        # quesFile    ='../data/miniCOCO_val_questions.json'
        # results is an array of objects of type {"answer": prediction (encoded), "question_id": id} and we must save it
        # to a json file that has an array of {"answer": prediction (plain text), "question_id": id}
        self.annFile = annFile
        self.quesFile = quesFile

        if already_decoded:
            decoded_results = results
        else:
            with open(ids_to_labels_file, 'r') as file:
                label_encoding = yaml.safe_load(file)

            decoded_results = list(map(lambda item: {"answer": label_encoding.get(item["answer"], "Unknown"),
                                                     "question_id": item["question_id"]}, results))

        with open(res_file, 'w') as file:
            json.dump(decoded_results, file, indent=4)

        self.resFile = res_file

    def print_accuracies(self):
        vqa = VQAhelper(self.annFile, self.quesFile)
        vqaRes = vqa.loadRes(self.resFile, self.quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        vqaEval.evaluate()
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        print("Per Question Type Accuracy is the following:")
        for quesType in vqaEval.accuracy['perQuestionType']:
            print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
        print("\n")
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

    def get_overall_accuracy(self):
        vqa = VQAhelper(self.annFile, self.quesFile)
        vqaRes = vqa.loadRes(self.resFile, self.quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        vqaEval.evaluate(vqaRes.getQuesIds())
        return vqaEval.accuracy['overall']

    def get_random_accuracy(self):

        pass

    def get_top_pics_accuracy(self):
        pass
