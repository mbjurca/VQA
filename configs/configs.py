from yacs.config import CfgNode as CN


_C = CN()

_C.DATASET = CN()
_C.DATASET.TRAIN_FILE = ""
_C.DATASET.TRAIN_IMG_FOLDER = ""
_C.DATASET.VAL_FILE = ""
_C.DATASET.VAL_IMG_FOLDER = ""
_C.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER = ""
_C.DATASET.LABELS = ""
_C.DATASET.WORD_VOCABULARY = ""
_C.DATASET.WORD_VOCABULARY_SIZE = 4476

_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 100
_C.TRAIN.LR = 0.001

_C.MODEL = CN()

_C.MODEL.TEXT_RNN = CN()
_C.MODEL.TEXT_RNN.SEQUENCE_LENGTH = 15
_C.MODEL.TEXT_RNN.HIDDEN_EMBEDDING_SIZE = 512
_C.MODEL.TEXT_RNN.INPUT_SIZE = 256

_C.MODEL.VIT = CN()
_C.MODEL.VIT.NO_IN_FEATURES = 2048
_C.MODEL.VIT.NO_OUT_FEATURES = 512
_C.MODEL.VIT.NO_PATHES = 36
_C.MODEL.VIT.NO_BLOCKS = 3
_C.MODEL.VIT.NO_HEADS = 8
_C.MODEL.VIT.DROPOUT = 0.1

_C.MODEL.SLM = CN()
_C.MODEL.SLM.NO_FEATURES= 512
_C.MODEL.SLM.SEQUENCE_LENGTH = 15
_C.MODEL.SLM.NO_BLOCKS = 3
_C.MODEL.SLM.NO_HEADS = 8
_C.MODEL.SLM.DROPOUT = 0.1


def update_configs(cfg, model_file_path, dataset_file_path, training_file_path):

    cfg.defrost()
    cfg.merge_from_file(model_file_path)
    cfg.merge_from_file(dataset_file_path)
    cfg.merge_from_file(training_file_path)
    cfg.freeze()

def get_configs():

    return _C.clone()