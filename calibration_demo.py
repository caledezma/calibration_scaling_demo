"""
Demonstration of model calibration, presented at nPlan's ML Paper Club.
"""
import pdb
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support
from numpy import argmax, log, array, exp

def simple_mlp(
    num_features,
    num_layers,
    num_units,
    reg_coeff,
    dropout_prob,
    num_targets,
    learning_rate,
):
    """Returns a MLP using the tf.keras functional API."""
    input_layer = tf.keras.layers.Input(shape=(num_features,))
    # Builds the MLP layers
    output = input_layer
    for _ in range(num_layers):
        output = tf.keras.layers.Dense(
            units=num_units,
            activation="relu",
            kernel_initializer="glorot_uniform",
            kernel_regularizer=tf.keras.regularizers.l2(l=reg_coeff))(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Dropout(rate=dropout_prob)(output)

    # Build the output layer
    output = tf.keras.layers.Dense(
        units=num_targets if num_targets > 2 else 1,
        activation="softmax",
        kernel_initializer="glorot_uniform",
    )(output)
    mlp_model = tf.keras.Model(input_layer, output)
    mlp_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            amsgrad=True,
        ),
        loss="categorical_crossentropy" if num_targets > 2 else "binary_crossentropy",
    )
    return mlp_model

class SigmoidLayer(tf.keras.layers.Layer):
    """Sigmoid layer with trainable parameters A and B:
        s(x) = 1 / (1 + exp(A*x + B)

        MVP: not working at the moment.
    """
    def __init__(self, prior_0, prior_1, **kwargs):
        self.prior_0 = prior_0
        self.prior_1 = prior_1
        super(SigmoidLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.A = self.add_weight(
            name="A",
            initializer=tf.keras.initializers.Constant(value=0),
            trainable=True,
        )
        self.B = self.add_weight(
            name="B",
            initializer=tf.keras.initializers.Constant(
                value=log((self.prior_0+1)/(self.prior_1+1))
            ),
            trainable=True,
        )
        self.built=True

    def call(self, x):
        return 1 / (1 + tf.keras.backend.exp(self.A * x + self.B))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], )

def platt_scaler(prior_0, prior_1, num_targets):
    """Returns a Keras model that does probability scaling. Relies on the SigmoidLayer defined
    above, so it's still not working"""
    input_layer = tf.keras.layers.Input(shape=(num_targets,))
    output_layer = SigmoidLayer(prior_0=prior_0, prior_1=prior_1)(input_layer)
    model = tf.keras.Model(input_layer, output_layer)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, amsgrad=True),
        loss="categorical_crossentropy" if num_targets > 2 else "binary_crossentropy"
    )
    return model

def platt_scaling(pred_proba, y_true, prior_0, prior_1):
    """Perform platt scaling on a single class. Returns A and B the parameters of a sigmoid.
    This is the algorithm presented in the original paper by Platt"""
    A = 0
    B = log((prior_0 + 1) / (prior_1 + 1))
    hi_target = (prior_1 + 1) / (prior_1 + 2)
    lo_target = 1 / (prior_0 + 1)
    lam = 1e-3
    old_err = 1e300
    pp = array([(prior_1 + 1) / (prior_0 + prior_1 + 2)] * pred_proba.size)
    count = 0
    for _ in tqdm(range(5000), desc="Fitting sigmoid"):
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        for idx in range(y_true.size):
            if y_true[idx]:
                t = hi_target
            else:
                t = lo_target
            d1 = pp[idx] - t
            d2 = pp[idx]*(1-pp[idx])
            a += pred_proba[idx] * pred_proba[idx] * d2
            b += d2
            c += pred_proba[idx] * d2
            d += pred_proba[idx] * d1
            e += d1
        if abs(d) < 1e-9 and abs(e) < 1e-9:
            return A, B
        old_A, old_B = A, B
        err = 0
        while True:
            det = (a + lam) * (b + lam) - c*c
            if (det == 0):
                lam *= 10
                continue
            A = old_A + ((b + lam) * d - c * e) / det
            B = old_B + ((a + lam) * e - c * d) / det
            err = 0
            for idx in range(y_true.size):
                p = 1 / (1 + exp(A*pred_proba[idx] + B))
                pp[idx] = p
                log_p = log(p) if p else -200
                log_1_p = log(1-p) if (1-p) else -200
                err -= t * log_p + (1-t) * log_1_p

            if err < old_err * (1+1e-7):
                lam *= 0.1
                break
            lam *= 10
            if lam > 1e6:
                break
        diff = err - old_err
        scale = 0.5 * (err + old_err + 1)
        if 1e-7*scale > diff > -1e-3 * scale:
            count += 1
        else:
            count = 0
        old_err = err
        if count == 3:
            return A, B
    return A, B

if __name__ == "__main__":
    """Runs the calibration demo"""
    iris_dataset = load_iris()
    pdb.set_trace()
    mlp = simple_mlp(
        num_features=iris_dataset["data"].shape[1],
        num_layers=4,
        num_units=50,
        reg_coeff=1e-3,
        dropout_prob=0.1,
        num_targets=3,
        learning_rate=1e-5,
    )
    train_features, test_features, train_targets, test_targets = train_test_split(
        iris_dataset["data"],
        tf.keras.utils.to_categorical(iris_dataset["target"]),
        test_size=0.3,
        stratify=tf.keras.utils.to_categorical(iris_dataset["target"]),
    )
    print("Fitting classifier")
    mlp.fit(
        x=train_features,
        y=train_targets,
        epochs=100,
        batch_size=8,
        verbose=0
    )
    y_pred = mlp.predict(x=test_features)
    p, r, f, _ = precision_recall_fscore_support(
        y_true=argmax(test_targets, axis=1),
        y_pred=argmax(y_pred, axis=1),
    )
    calib_curves = []
    for idx in range(test_targets.shape[1]):
        f1 = f1_score(
            y_true=test_targets[:, idx],
            y_pred=(argmax(y_pred, axis=1) == idx).astype(int),
            average="binary",
        )
        true_prob, pred_prob = calibration_curve(
            y_true=test_targets[:, idx],
            y_prob=y_pred[:, idx],
            n_bins=5
        )
        fig = plt.figure()
        plt.plot(pred_prob, true_prob, marker="o", label="classifier")
        plt.plot([0,1], [0,1], ":", label="Perfect calibration")
        plt.title("Classifier calibration for {0}. F1={1:.2%}.".format(
            iris_dataset["target_names"][idx],
            f1,)
        )
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.legend()
        plt.savefig("calibration_curve_{0}.pdf".format(idx))
        plt.close(fig)

    print("Calibrating")
    for idx in range(test_targets.shape[1]):
        A, B = platt_scaling(
            pred_proba=y_pred[:, idx],
            y_true=test_targets[:, idx],
            prior_0=test_targets[:, idx].size - test_targets[:, idx].sum(),
            prior_1=test_targets[:, idx].sum(),
        )
        calib_y = (1 / (1 + exp(A * y_pred[:, idx] + B)))
        true_prob, pred_prob = calibration_curve(
            y_true=test_targets[:, idx],
            y_prob=calib_y,
            n_bins=5
        )
        fig = plt.figure()
        plt.plot(pred_prob, true_prob, marker="o", label="classifier")
        plt.plot([0,1], [0,1], ":", label="Perfect calibration")
        plt.title("Calibrated classifier for {0}".format(iris_dataset["target_names"][idx]))
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        plt.legend()
        plt.savefig("post_calibration_curve_{0}.pdf".format(idx))
        plt.close(fig)

    # Scaling using the Keras layer is not working very well, more testing needed
    # calib_scaler = platt_scaler(
    #     prior_0=test_targets[:, idx].size - test_targets[:, idx].sum(),
    #     prior_1=test_targets[:, idx].sum(),
    #     num_targets=test_targets.shape[1],
    # )
    # calib_scaler.fit(
    #     x=y_pred,
    #     y=test_targets,
    #     epochs=1000,
    #     batch_size=8,
    #     verbose=0
    # )
    # calib_y = calib_scaler.predict(y_pred)