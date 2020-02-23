import os
import pandas as pd
from multiprocessing import cpu_count
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .utils import get_model_weights_path, get_model_json_path, get_model_history_path


def train_model(model, train, test, epochs=1000, path="./models"):
    weights_path = get_model_weights_path(model, path)
    json_path = get_model_json_path(model, path)

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Old Weights loaded from {}".format(weights_path))

    with open(json_path, "w") as f:
        f.write(model.to_json())
    
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
        workers=cpu_count()//2,
        use_multiprocessing=True
    ).history

    pd.DataFrame(
        history
    ).to_csv(get_model_history_path(model, path))

    return model