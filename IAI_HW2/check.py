"""Check your implementation of `layers.py` and `losses.py`.

Usage:
    python check.py [--seed 0] [--n 20] [--batch_size 16]

1. Layer checks: the forward pass of every layer and loss is compared with
   PyTorch, and the backward pass is compared with a numerical gradient.
2. Network checks: the gradients of the public networks are compared with
   PyTorch, using the same error and score table as the grading.

The grading scripts also use the functions in this file.
"""
import argparse
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np


PUBLIC_NETWORKS = [
    {
        "name": "multiclass",
        "dims": [784, 256, 128, 10],
        "activations": ["relu", "relu", "softmax"],
        "loss": "cross_entropy",
    },
    {
        "name": "binary",
        "dims": [100, 64, 32, 1],
        "activations": ["relu", "relu", "sigmoid"],
        "loss": "binary_cross_entropy",
    },
]

# layer class of each activation; "none" means no activation
ACTIVATIONS = {"relu": "ReLU", "sigmoid": "Sigmoid", "softmax": "Softmax", "none": None}
# loss class of each loss
LOSSES = {"cross_entropy": "CrossEntropy", "binary_cross_entropy": "BinaryCrossEntropy"}

# (upper bound of the error, score), checked in order
SCORE_TABLE = [(1e-4, 100), (1e-3, 90), (1e-2, 80)]
# a check passes if its error is below the bound for 100 points
PASS_THRESHOLD = SCORE_TABLE[0][0]


# ---------------------------------------------------------------- networks

def validate_network(spec):
    """Raise `ValueError` if `spec` is not a valid network."""
    name, dims = spec["name"], spec["dims"]
    activations, loss = spec["activations"], spec["loss"]
    if len(dims) < 2 or len(activations) != len(dims) - 1:
        raise ValueError(f"{name}: need one activation per linear layer")
    unknown = [a for a in activations if a not in ACTIVATIONS]
    if unknown:
        raise ValueError(f"{name}: unknown activations {unknown}")
    if loss not in LOSSES:
        raise ValueError(f"{name}: unknown loss {loss!r}")
    if loss == "cross_entropy" and activations[-1] != "softmax":
        raise ValueError(f"{name}: cross_entropy needs a softmax output")
    if loss == "binary_cross_entropy" and (activations[-1] != "sigmoid" or dims[-1] != 1):
        raise ValueError(f"{name}: binary_cross_entropy needs a single sigmoid output")


def param_names(spec):
    """Names of the parameters, in the order of `Sequential.params()`."""
    names = []
    for i in range(1, len(spec["dims"])):
        names += [f"W{i}", f"b{i}"]
    return names


def components(spec):
    """Names of the layer and loss classes used by a network."""
    names = {"MatMul", "Bias", LOSSES[spec["loss"]]}
    names.update(ACTIVATIONS[a] for a in spec["activations"] if a != "none")
    return names


def generate_data(spec, seed, n, batch_size):
    """Generate the parameters and the batches of a network.

    Returns:
        params (`List[np.ndarray]`):
            `[W1, b1, W2, b2, ...]`.
        batches (`List[Tuple[np.ndarray, np.ndarray, np.ndarray]]`):
            `n` triples of inputs with shape `(batch_size, input_dim)`,
            labels, and sample weights with shape `(batch_size,)`.
    """
    rng = np.random.default_rng(seed)
    dims = spec["dims"]
    params = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        params.append(rng.standard_normal((d_in, d_out)) / np.sqrt(d_in))
        params.append(rng.standard_normal(d_out) * 0.1)
    batches = []
    for _ in range(n):
        x = rng.random((batch_size, dims[0]))
        if spec["loss"] == "cross_entropy":
            y = rng.integers(0, dims[-1], batch_size)
        else:
            y = rng.integers(0, 2, (batch_size, 1)).astype(np.float64)
        # the loss is a weighted mean, so a loss must use the grad it receives
        w = rng.uniform(0.5, 1.5, batch_size)
        batches.append((x, y, w))
    return params, batches


def numpy_gradients(spec, layers, losses, seed, n, batch_size):
    """Gradients of a network built from the `layers` and `losses` modules.

    Returns:
        grads (`Dict[str, np.ndarray]`):
            gradient of every parameter, stacked over the batches into shape
            `(n, *param.shape)`.
    """
    params, batches = generate_data(spec, seed, n, batch_size)
    names = param_names(spec)

    # model initialization
    model = layers.Sequential()
    for i, activation in enumerate(spec["activations"]):
        model.append(layers.MatMul(params[2 * i].copy()))
        model.append(layers.Bias(params[2 * i + 1].copy()))
        if activation != "none":
            model.append(getattr(layers, ACTIVATIONS[activation])())
    loss_fn = getattr(losses, LOSSES[spec["loss"]])()

    grads = {name: [] for name in names}
    for x, y_true, w in batches:
        # forward
        y_pred = model.forward(x)
        loss = loss_fn.forward(y_pred, y_true)
        if np.shape(loss) != (batch_size,):
            raise ValueError(
                f"{type(loss_fn).__name__}.forward returned shape "
                f"{np.shape(loss)}, expected {(batch_size,)}")

        # backward
        grad = w / batch_size
        model.backward(loss_fn.backward(grad))

        # collect gradients; copy them in case a layer reuses its arrays
        model_grads = model.grads()
        if len(model_grads) != len(names):
            raise ValueError(
                f"grads() returned {len(model_grads)} arrays, expected {len(names)}")
        for name, param, g in zip(names, params, model_grads):
            g = np.array(g, dtype=np.float64)
            if g.shape != param.shape:
                raise ValueError(
                    f"{name}: gradient shape {g.shape} != parameter shape {param.shape}")
            grads[name].append(g)
    return {name: np.stack(g) for name, g in grads.items()}


def torch_activation(name, x):
    """Apply the activation `name` to the tensor `x` with PyTorch."""
    import torch

    if name == "relu":
        return torch.relu(x)
    if name == "sigmoid":
        return torch.sigmoid(x)
    if name == "softmax":
        return torch.softmax(x, dim=-1)
    return x


def torch_loss(name, y_pred, y_true):
    """Loss of every sample with PyTorch, written from its formula."""
    import torch

    if name == "cross_entropy":
        y_true = torch.as_tensor(y_true, dtype=torch.int64)
        return -torch.log(y_pred[torch.arange(len(y_true)), y_true])
    y_true = torch.as_tensor(y_true, dtype=torch.float64)
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss.mean(dim=-1)


def torch_gradients(spec, seed, n, batch_size):
    """Reference gradients of a network, computed by PyTorch in float64.

    Returns the same structure as `numpy_gradients`.
    """
    import torch

    params, batches = generate_data(spec, seed, n, batch_size)
    names = param_names(spec)
    tensors = [torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in params]

    grads = {name: [] for name in names}
    for x, y_true, w in batches:
        # forward
        h = torch.tensor(x, dtype=torch.float64)
        for i, activation in enumerate(spec["activations"]):
            h = torch_activation(activation, h @ tensors[2 * i] + tensors[2 * i + 1])
        loss = torch_loss(spec["loss"], h, y_true)

        # backward
        weighted = (loss * torch.as_tensor(w)).sum() / batch_size
        batch_grads = torch.autograd.grad(weighted, tensors)
        for name, g in zip(names, batch_grads):
            grads[name].append(g.numpy())
    return {name: np.stack(g) for name, g in grads.items()}


# ---------------------------------------------------------------- errors

def tensor_error(g, ref):
    """Largest absolute difference, relative to the largest reference value."""
    scale = np.abs(ref).max()
    error = np.abs(g - ref).max() / (scale if scale > 0 else 1.0)
    return float(error) if np.isfinite(error) else float("inf")


def gradient_errors(grads, reference):
    """Error of every parameter: the largest `tensor_error` over the batches."""
    errors = {}
    for name, ref in reference.items():
        g = grads[name]
        if g.shape != ref.shape:
            raise ValueError(f"{name}: shape {g.shape} != reference shape {ref.shape}")
        errors[name] = max(tensor_error(g_batch, ref_batch)
                           for g_batch, ref_batch in zip(g, ref))
    return errors


def diff2score(error):
    """Score of a network from its error."""
    for bound, score in SCORE_TABLE:
        if error < bound:
            return score
    return 0


# ---------------------------------------------------------------- layer checks

COMPONENTS = ["MatMul", "Bias", "ReLU", "Sigmoid", "Softmax",
              "CrossEntropy", "BinaryCrossEntropy"]


class CheckFailed(Exception):
    """A layer check found a wrong value or a wrong shape."""


def numerical_gradient(f, x, h=1e-6):
    """Central difference gradient of the scalar function `f()` w.r.t. `x`.

    Every element of `x` is changed in place while `f` is evaluated, and is
    restored afterwards.
    """
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        value = x[i]
        x[i] = value + h
        f_plus = f()
        x[i] = value - h
        f_minus = f()
        x[i] = value
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


def check_close(what, value, expected):
    """Raise `CheckFailed` unless `value` matches `expected`."""
    value = np.asarray(value, dtype=np.float64)
    if value.shape != expected.shape:
        raise CheckFailed(f"{what}: shape {value.shape}, expected {expected.shape}")
    error = tensor_error(value, expected)
    if not error < PASS_THRESHOLD:
        raise CheckFailed(f"{what} error {error:.1e}")


def check_layer(layer, x, out_shape, expected, grad_names):
    """Check the forward and backward pass of `layer` at `x`.

    Parameters:
        out_shape (`Tuple[int, ...]`):
            expected shape of the output.
        expected (`np.ndarray` or `None`):
            expected output computed by PyTorch; `None` skips the comparison.
        grad_names (`List[str]`):
            names of the parameter gradients, e.g. `["W_grad"]`.
    """
    rng = np.random.default_rng(1)
    out = np.asarray(layer.forward(x.copy()))
    if out.shape != out_shape:
        raise CheckFailed(f"forward: shape {out.shape}, expected {out_shape}")
    if expected is not None:
        check_close("forward", out, expected)

    # backward at the unperturbed input
    g = rng.standard_normal(out_shape)
    x_grad = np.array(layer.backward(g), dtype=np.float64)
    params = layer.params()
    grads = [np.array(p, dtype=np.float64) for p in layer.grads()]
    if len(grads) != len(params):
        raise CheckFailed(
            f"grads() returned {len(grads)} arrays, params() returned {len(params)}")

    # numerical gradients of sum(g * forward(x))
    def f():
        return np.sum(g * layer.forward(x.copy()))

    check_close("backward: input gradient", x_grad, numerical_gradient(f, x))
    for name, param, grad in zip(grad_names, params, grads):
        check_close(f"backward: {name}", grad, numerical_gradient(f, param))


def check_loss(loss_fn, y_pred, y_true, expected):
    """Check the forward and backward pass of `loss_fn` at `(y_pred, y_true)`."""
    rng = np.random.default_rng(2)
    batch_size = y_pred.shape[0]
    out = np.asarray(loss_fn.forward(y_pred.copy(), y_true))
    if out.shape != (batch_size,):
        raise CheckFailed(f"forward: shape {out.shape}, expected {(batch_size,)}")
    if expected is not None:
        check_close("forward", out, expected)

    # backward with a random upstream gradient
    w = rng.uniform(0.5, 1.5, batch_size)
    grad = np.array(loss_fn.backward(w), dtype=np.float64)

    # numerical gradient of sum(w * forward(y_pred))
    def f():
        return np.sum(w * loss_fn.forward(y_pred.copy(), y_true))

    check_close("backward", grad, numerical_gradient(f, y_pred))


def describe_exception(e):
    """`Type: message (file:line)`, pointing at `layers.py` or `losses.py` if possible."""
    where = ""
    for frame in traceback.extract_tb(e.__traceback__):
        if Path(frame.filename).name in ("layers.py", "losses.py"):
            where = f" ({Path(frame.filename).name}:{frame.lineno})"
    return f"{type(e).__name__}: {e}{where}"


def layer_checks(layers, losses, torch=None, seed=0):
    """Check every layer and loss on small random inputs.

    Parameters:
        torch (`module` or `None`):
            PyTorch, used by the forward checks; `None` skips them.

    Returns:
        results (`List[Tuple[str, str, str]]`):
            `(component, status, detail)` in the order of `COMPONENTS`, where
            status is `PASS`, `FAIL`, `NOT IMPLEMENTED` or `ERROR`.
    """
    rng = np.random.default_rng(seed)
    x_in = rng.standard_normal((3, 4))
    W = rng.standard_normal((4, 5))
    b = rng.standard_normal(5)
    x = rng.standard_normal((3, 5))
    x_relu = x + 0.1 * np.sign(x)            # keep away from the kink at 0
    probs = rng.random((3, 5)) + 0.1
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 5, 3)
    p_binary = rng.uniform(0.1, 0.9, (3, 1))
    y_binary = np.array([[1.0], [0.0], [1.0]])

    def expected(fn, *arrays):
        if torch is None:
            return None
        return fn(*[torch.as_tensor(a) for a in arrays]).numpy()

    checks = {
        "MatMul": lambda: check_layer(
            layers.MatMul(W.copy()), x_in, (3, 5),
            expected(lambda x, W: x @ W, x_in, W), ["W_grad"]),
        "Bias": lambda: check_layer(
            layers.Bias(b.copy()), x, (3, 5),
            expected(lambda x, b: x + b, x, b), ["b_grad"]),
        "ReLU": lambda: check_layer(
            layers.ReLU(), x_relu, (3, 5),
            expected(lambda x: torch_activation("relu", x), x_relu), []),
        "Sigmoid": lambda: check_layer(
            layers.Sigmoid(), x, (3, 5),
            expected(lambda x: torch_activation("sigmoid", x), x), []),
        "Softmax": lambda: check_layer(
            layers.Softmax(), x, (3, 5),
            expected(lambda x: torch_activation("softmax", x), x), []),
        "CrossEntropy": lambda: check_loss(
            losses.CrossEntropy(), probs, labels,
            expected(lambda p, t: torch_loss("cross_entropy", p, t), probs, labels)),
        "BinaryCrossEntropy": lambda: check_loss(
            losses.BinaryCrossEntropy(), p_binary, y_binary,
            expected(lambda p, y: torch_loss("binary_cross_entropy", p, y),
                     p_binary, y_binary)),
    }

    results = []
    for name in COMPONENTS:
        try:
            checks[name]()
        except NotImplementedError:
            results.append((name, "NOT IMPLEMENTED", ""))
        except CheckFailed as e:
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            results.append((name, "ERROR", describe_exception(e)))
        else:
            detail = "" if torch is not None else "forward not checked (no PyTorch)"
            results.append((name, "PASS", detail))
    return results


# ---------------------------------------------------------------- command line

def network_checks(layers, losses, results, torch, seed, n, batch_size):
    """Compare the gradients of the public networks with PyTorch.

    Parameters:
        results (`List[Tuple[str, str, str]]`):
            output of `layer_checks`; networks that use an unimplemented
            component are skipped.

    Returns:
        rows (`List[Dict]`):
            `name`, `status` (`OK`, `SKIPPED` or `ERROR`), `error`, `score`
            and `detail` of every public network.
    """
    status = {name: s for name, s, _ in results}
    rows = []
    for spec in PUBLIC_NETWORKS:
        row = {"name": spec["name"], "status": "OK", "error": None, "score": 0, "detail": ""}
        missing = [c for c in COMPONENTS
                   if c in components(spec) and status[c] == "NOT IMPLEMENTED"]
        if torch is None:
            row.update(status="SKIPPED", detail="PyTorch is not installed")
        elif missing:
            row.update(status="SKIPPED", detail="not implemented: " + ", ".join(missing))
        else:
            try:
                grads = numpy_gradients(spec, layers, losses, seed, n, batch_size)
            except Exception as e:
                row.update(status="ERROR", detail=describe_exception(e))
            else:
                errors = gradient_errors(grads, torch_gradients(spec, seed, n, batch_size))
                wrong = [name for name, error in errors.items() if not error < PASS_THRESHOLD]
                row["error"] = max(errors.values())
                row["score"] = diff2score(row["error"])
                if wrong:
                    row["detail"] = "wrong: " + ", ".join(wrong)
        rows.append(row)
    return rows


def import_student_modules(directory):
    """Import `layers.py` and `losses.py` from `directory` without PyTorch.

    The files are loaded by path and registered as the modules `layers` and
    `losses`. `import torch` fails inside them, as in the grading environment.

    Returns:
        modules (`Tuple[module, module]`):
            the `layers` and `losses` modules.
    """
    blocked = "torch" not in sys.modules
    if blocked:
        sys.modules["torch"] = None     # makes `import torch` raise ImportError
    try:
        modules = []
        for name in ("layers", "losses"):
            spec = importlib.util.spec_from_file_location(name, Path(directory) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[name]
                raise
            modules.append(module)
        return tuple(modules)
    finally:
        if blocked:
            del sys.modules["torch"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--n', default=20, type=int, help='number of batches per network')
    parser.add_argument('--batch_size', default=16, type=int)
    args = parser.parse_args()

    # import layers.py and losses.py without PyTorch, as in the grading
    try:
        layers, losses = import_student_modules(Path(__file__).resolve().parent)
    except ImportError as e:
        if e.name == "torch":
            print("layers.py or losses.py imports torch, but only numpy is "
                  "available in the grading environment.")
        else:
            print(f"Cannot import layers.py and losses.py: {describe_exception(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Cannot import layers.py and losses.py: {describe_exception(e)}")
        sys.exit(1)

    try:
        import torch
    except ImportError:
        torch = None
        print("PyTorch is not installed: the forward checks and the network "
              "checks are skipped.\n")

    print("Layer checks (forward vs PyTorch, backward vs numerical gradient)")
    results = layer_checks(layers, losses, torch, args.seed)
    for name, status, detail in results:
        print(f"  {name:<20}{status:<17}{detail}".rstrip())

    print()
    print(f"Network checks vs PyTorch (seed={args.seed}, n={args.n}, "
          f"batch_size={args.batch_size})")
    rows = network_checks(layers, losses, results, torch, args.seed, args.n, args.batch_size)
    for row in rows:
        if row["status"] == "OK":
            line = f"error {row['error']:.1e}    score {row['score']:<4} {row['detail']}"
        else:
            line = f"{row['status']:<17}{row['detail']}"
        print(f"  {row['name']:<20}{line}".rstrip())

    passed = (all(status == "PASS" for _, status, _ in results)
              and all(row["status"] == "OK" and row["score"] == 100 for row in rows))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
