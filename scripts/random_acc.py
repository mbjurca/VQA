import sys

sys.path.append('../configs/')
sys.path.append('../datasets/')
sys.path.append('../utils/')
sys.path.append('../lib/')

from configs import update_configs, get_configs
from coco_vqa_dataset import VQA_dataset
from Evaluator import Evaluator
import random

DATASET_CFG_FILE = "../configs/dataset.yaml"
MODEL_CFG_FILE = "../configs/model.yaml"
TRAIN_CFG_FILE = "../configs/train.yaml"


def compute_acc(dataset, cfg, subset='train'):
    k = 1000
    ids = dataset.get_question_ids()
    top_answers = dataset.get_top_k_answers(k)
    top_answers = list(top_answers.keys())

    result_list = []
    for question_id in ids:
        result_list.append({
            "answer": random.choice(top_answers),
            "question_id": question_id
        })

    print(f'Performance for randomly selected answers out of the top {k} answers for the {subset} subset:')
    if subset == 'train':
        evaluator = Evaluator(cfg.TRAIN.ANNOTATIONS_FILE, cfg.TRAIN.QUESTIONS_FILE, cfg.DATASET.TRAIN_IDS_TO_LABELS,
                              result_list, already_decoded=True, res_file=f'{subset}_random.json')
    else:
        evaluator = Evaluator(cfg.VAL.ANNOTATIONS_FILE, cfg.VAL.QUESTIONS_FILE, cfg.DATASET.TRAIN_IDS_TO_LABELS,
                              result_list, already_decoded=True, res_file=f'{subset}_random.json')
    # evaluator.print_accuracies()
    print(f'overall accuracy = {evaluator.get_overall_accuracy()}')

    result_list = []
    for question_id in ids:
        result_list.append({
            "answer": top_answers[0],
            "question_id": question_id
        })

    print(f'Performance when most common answer is always selected for the {subset} subset:')
    if subset == 'train':
        evaluator = Evaluator(cfg.TRAIN.ANNOTATIONS_FILE, cfg.TRAIN.QUESTIONS_FILE, cfg.DATASET.TRAIN_IDS_TO_LABELS,
                              result_list, already_decoded=True, res_file=f'{subset}_most_common.json')
    else:
        evaluator = Evaluator(cfg.VAL.ANNOTATIONS_FILE, cfg.VAL.QUESTIONS_FILE, cfg.DATASET.TRAIN_IDS_TO_LABELS,
                              result_list, already_decoded=True, res_file=f'{subset}_most_common.json')
    # evaluator.print_accuracies()
    print(f'overall accuracy = {evaluator.get_overall_accuracy()}')


if __name__ == '__main__':
    # create the config
    cfg = get_configs()
    update_configs(cfg, MODEL_CFG_FILE, DATASET_CFG_FILE, TRAIN_CFG_FILE)

    train_dataset = VQA_dataset(dataset_file=cfg.DATASET.TRAIN_FILE,
                                labels_to_ids_file=cfg.DATASET.TRAIN_LABELS_TO_IDS,
                                ids_to_labels_file=cfg.DATASET.TRAIN_IDS_TO_LABELS,
                                vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER,
                                token_type=cfg.MODEL.TEXT.TOKEN_TYPE)
    compute_acc(train_dataset, cfg)

    validation_dataset = VQA_dataset(dataset_file=cfg.DATASET.VAL_FILE,
                                     labels_to_ids_file=cfg.DATASET.TRAIN_LABELS_TO_IDS,
                                     ids_to_labels_file=cfg.DATASET.TRAIN_IDS_TO_LABELS,
                                     vocabulary_file=cfg.DATASET.WORD_VOCABULARY,
                                     image_embedding_folder=cfg.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER,
                                     token_type=cfg.MODEL.TEXT.TOKEN_TYPE)
    compute_acc(validation_dataset, cfg, 'validation')
