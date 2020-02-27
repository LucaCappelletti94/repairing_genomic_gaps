import os
import pandas as pd
from multiprocessing import cpu_count
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .utils import get_model_weights_path, get_model_json_path, get_model_history_path
from plot_keras_history import chain_histories


def train_model(model, train, test, epochs=1000, path="./models"):
    weights_path = get_model_weights_path(model, path)
    json_path = get_model_json_path(model, path)
    history_path = get_model_history_path(model, path)

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    old_history = None
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        old_history = pd.read_csv(history_path)
        print("Old Weights loaded from {}".format(weights_path))

    with open(json_path, "w") as f:
        f.write(model.to_json())
    
    history = model.fit_generator(
        train,
        steps_per_epoch=train.steps_per_epoch,
        epochs=epochs,
        shuffle=True,
        verbose=1,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=0,
                mode='min',
                restore_best_weights=True
            ),
            ModelCheckpoint(
                weights_path,
                monitor='val_loss',
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode='min'
            )
        ],
        validation_data=test,
        validation_steps=test.steps_per_epoch,
        workers=max(1, min(3, cpu_count()//2)),
        use_multiprocessing=True
    ).history

    pd.DataFrame(chain_histories(
        history,
        old_history
    )).to_csv(history_path, index=False)

    return model