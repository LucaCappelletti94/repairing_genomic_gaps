import os
import pandas as pd
from multiprocessing import cpu_count
from plot_keras_history import plot_history
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model(model, train, test, epochs=1000, path="./models"):
    saved_weights_path = "{path}/{name}/best_weights_{name}.hdf5".format(path=path, name=model.name)

    os.makedirs(os.path.dirname(saved_weights_path), exist_ok=True)
    if os.path.exists(saved_weights_path):
        model.load_weights(saved_weights_path)
        print("Old Weights loaded from {}".format(saved_weights_path))

    model_json = model.to_json()
    with open("{path}/{name}/model_{name}.json".format(path=path, name=model.name), "w") as json_file:
        json_file.write(model_json)

    
    history = model.fit_generator(
        train,
        steps_per_epoch=train.steps_per_epoch/15,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=5,
                verbose=0,
                mode='min',
                restore_best_weights=True
            ),
            ModelCheckpoint(
                saved_weights_path,
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='min'
            )
        ],
        validation_data=test,
        validation_steps=test.steps_per_epoch,
        workers=cpu_count()//2,
        use_multiprocessing=True
    ).history
    pd.DataFrame(
        history
    ).to_csv("{path}/{name}/history_{name}.csv".format(path=path, name=model.name))
    plot_history(history, path="{path}/{name}/history_{name}.png".format(path=path, name=model.name))
    return model