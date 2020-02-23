from tensorflow.keras import Model


def get_model_weights_path(model: Model, path: str = "./models") -> str:
    return "{path}/{name}/best_weights_{name}.hdf5".format(path=path, name=model.name)


def get_model_json_path(model: Model, path: str = "./models") -> str:
    return "{path}/{name}/model_{name}.json".format(path=path, name=model.name)


def get_model_history_path(model: Model, path: str = "./models") -> str:
    return "{path}/{name}/history_{name}.csv".format(path=path, name=model.name)
