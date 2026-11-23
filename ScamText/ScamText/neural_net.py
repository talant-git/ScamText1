"""AI Fundamentals, 9-апта: Perceptrons / Neural Networks.

A small neural network written from scratch with numpy (forward pass, backpropagation, Adam):
    input (TF-IDF vector) -> hidden layer (tanh) -> output (sigmoid) = P(Spam)
Also contains the classic single-layer perceptron, used to show why XOR needs a hidden layer.
"""
import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class MLP:
    def __init__(self, n_in, hidden=16, lr=0.02, l2=1e-4, init_std=0.1, seed=42):
        rng = np.random.default_rng(seed)
        self.hidden, self.lr, self.l2 = hidden, lr, l2
        self.W1 = rng.normal(0, init_std, (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 1.0 / np.sqrt(hidden), hidden)
        self.b2 = 0.0
        self.loss_history = []

    # ---- forward ------------------------------------------------------------------
    def forward(self, X):
        h = np.tanh(X @ self.W1 + self.b1)          # hidden activations
        p = _sigmoid(h @ self.W2 + self.b2)         # P(Spam)
        return h, p

    def predict_proba(self, X):
        return self.forward(X)[1]

    # ---- loss and backpropagation ------------------------------------------------------
    def loss_and_grads(self, X, y, sw=None):
        n = X.shape[0]
        sw = np.ones(n) if sw is None else sw
        h, p = self.forward(X)
        eps = 1e-12
        data_loss = -np.sum(sw * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / np.sum(sw)
        reg = 0.5 * self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        d_out = sw * (p - y) / np.sum(sw)           # dLoss/d(output pre-activation)
        gW2 = h.T @ d_out + self.l2 * self.W2
        gb2 = float(np.sum(d_out))
        d_h = np.outer(d_out, self.W2) * (1 - h ** 2)   # back through tanh
        gW1 = np.asarray(X.T @ d_h) + self.l2 * self.W1
        gb1 = d_h.sum(axis=0)
        return data_loss + reg, (gW1, gb1, gW2, gb2)

    # ---- training: full-batch Adam ------------------------------------------------------
    def fit(self, X, y, epochs=100, sample_weight=None, X_val=None, y_val=None, record_every=1):
        y = np.asarray(y, dtype=float)
        params = [self.W1, self.b1, self.W2, np.array(self.b2)]
        m = [np.zeros_like(p, dtype=float) for p in params]
        v = [np.zeros_like(p, dtype=float) for p in params]
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        self.loss_history, self.val_loss_history = [], []
        for t in range(1, epochs + 1):
            loss, grads = self.loss_and_grads(X, y, sample_weight)
            grads = [grads[0], grads[1], grads[2], np.array(grads[3])]
            for i, g in enumerate(grads):
                m[i] = b1c * m[i] + (1 - b1c) * g
                v[i] = b2c * v[i] + (1 - b2c) * g * g
                step = self.lr * (m[i] / (1 - b1c ** t)) / (np.sqrt(v[i] / (1 - b2c ** t)) + eps)
                params[i] -= step
            self.b2 = float(params[3])
            if t % record_every == 0 or t == epochs:
                self.loss_history.append(float(loss))
                if X_val is not None:
                    p = np.clip(self.predict_proba(X_val), 1e-12, 1 - 1e-12)
                    self.val_loss_history.append(float(-np.mean(y_val * np.log(p) + (1 - y_val) * np.log(1 - p))))
        return self

    # ---- persistence (numpy .npz, no pickle) -------------------------------------------------
    def save(self, path):
        np.savez_compressed(path, W1=self.W1.astype(np.float32), b1=self.b1, W2=self.W2, b2=np.array(self.b2),
                            hidden=np.array(self.hidden), loss=np.array(self.loss_history))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(d["W1"].shape[0], int(d["hidden"]))
        m.W1, m.b1, m.W2, m.b2 = d["W1"].astype(np.float64), d["b1"], d["W2"], float(d["b2"])
        m.loss_history = d["loss"].tolist()
        return m


class Perceptron:
    """Rosenblatt perceptron: step activation, update only on mistakes. Linear decision boundary."""

    def __init__(self, n_in, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.w, self.b, self.lr = rng.normal(0, 0.5, n_in), 0.0, lr

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def fit(self, X, y, epochs=100):
        self.errors = []
        for _ in range(epochs):
            wrong = 0
            for xi, yi in zip(X, y):
                err = yi - int(xi @ self.w + self.b > 0)
                if err:
                    self.w += self.lr * err * xi
                    self.b += self.lr * err
                    wrong += 1
            self.errors.append(wrong)
        return self


XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_Y = np.array([0, 1, 1, 0])


def xor_demo(hidden=4, lr=0.05, epochs=1000, seed=1):
    """Train a perceptron and a neural network on XOR. The perceptron cannot learn it (not linearly separable)."""
    per = Perceptron(2, lr=0.1, seed=seed).fit(XOR_X, XOR_Y, epochs=min(epochs, 1000))
    net = MLP(2, hidden=hidden, lr=lr, l2=0.0, init_std=1.0, seed=seed)
    net.fit(XOR_X, XOR_Y, epochs=epochs, record_every=max(1, epochs // 40))
    pp, np_ = per.predict(XOR_X), (net.predict_proba(XOR_X) >= 0.5).astype(int)
    return {
        "perceptron": {"accuracy": float((pp == XOR_Y).mean()), "predictions": pp.tolist(),
                       "errors_last_epoch": int(per.errors[-1])},
        "neural_net": {"accuracy": float((np_ == XOR_Y).mean()), "predictions": np_.tolist(),
                       "probabilities": [round(float(x), 3) for x in net.predict_proba(XOR_X)],
                       "loss_curve": [round(x, 4) for x in net.loss_history]},
        "inputs": XOR_X.astype(int).tolist(), "targets": XOR_Y.tolist(),
    }
