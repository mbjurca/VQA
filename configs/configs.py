from yacs.config import CfgNode as CN


_C = CN()

_C.DATASET = CN()
_C.DATASET.TRAIN_FILE = ""
_C.DATASET.TRAIN_IMG_FOLDER = ""
_C.DATASET.VAL_FILE = ""
_C.DATASET.VAL_IMG_FOLDER = ""
_C.DATASET.TRAIN_VAL_IMG_EMBEDDINGS_FOLDER = ""

_C.TRAIN = CN()


def update_configs(cfg, model_file_path, dataset_file_path, training_file_path):

    cfg.defrost()
    cfg.merge_from_file(model_file_path)
    cfg.merge_from_file(dataset_file_path)
    cfg.merge_from_file(training_file_path)
    cfg.freeze()

def get_configs():

    return _C.clone()