import argparse
import os

import jax
import jax.numpy as jnp
from jax import lax, random

os.environ.setdefault("JAX_PLATFORMS", "mps")


def run_benchmark(unroll=1):
    print(f"Running benchmark with unroll={unroll}")

    input_dim = 128
    hidden_dim = 128
    num_layers = 4  # 4-layer Stacked LSTM
    seq_len = 10000

    key = random.PRNGKey(42)
    keys = random.split(key, 20)

    # LSTM Weights
    W_lstm = [
        random.normal(keys[i], (hidden_dim, 4 * hidden_dim)) for i in range(num_layers)
    ]
    U_lstm = [
        random.normal(keys[i + 4], (hidden_dim, 4 * hidden_dim))
        for i in range(num_layers)
    ]
    b_lstm = [random.normal(keys[i + 8], (4 * hidden_dim,)) for i in range(num_layers)]

    # MLP Weights
    W_mlp = [
        random.normal(keys[12], (hidden_dim, hidden_dim)),
        random.normal(keys[13], (hidden_dim, hidden_dim)),
        random.normal(keys[14], (hidden_dim, hidden_dim)),
    ]
    b_mlp = [
        random.normal(keys[15], (hidden_dim,)),
        random.normal(keys[16], (hidden_dim,)),
        random.normal(keys[17], (hidden_dim,)),
    ]

    def process(init_h, init_c, xs):
        def body(states, x):
            h_list, c_list = states
            new_h_list = []
            new_c_list = []
            curr_in = x

            # 4-layer stacked LSTM sequence
            for i in range(num_layers):
                h = h_list[i]
                c = c_list[i]

                proj = curr_in @ W_lstm[i] + h @ U_lstm[i] + b_lstm[i]
                i_part, f_part, o_part, g_part = jnp.split(proj, 4, axis=-1)

                i_gate = jax.nn.sigmoid(i_part)
                f_gate = jax.nn.sigmoid(f_part)
                o_gate = jax.nn.sigmoid(o_part)
                g_gate = jnp.tanh(g_part)

                new_c = f_gate * c + i_gate * g_gate
                new_h = o_gate * jnp.tanh(new_c)

                new_h_list.append(new_h)
                new_c_list.append(new_c)
                curr_in = new_h

            # 3-layer MLP Readout
            out = curr_in
            for i in range(3):
                out = jax.nn.relu(out @ W_mlp[i] + b_mlp[i])

            return (new_h_list, new_c_list), out

        if unroll == 0:
            h, c = init_h, init_c
            outs = []
            for i in range(seq_len):
                (h, c), out = body((h, c), xs[i])
                outs.append(out)
            return (h, c), jnp.stack(outs)
        else:
            return lax.scan(body, (init_h, init_c), xs, unroll=unroll)

    @jax.jit
    def process_grad(init_h, init_c, xs):
        def loss_fn(xs_):
            _, outs = process(init_h, init_c, xs_)
            return jnp.sum(outs)

        return jax.grad(loss_fn)(xs)

    # Initialize states
    init_h = [jnp.zeros((hidden_dim,), dtype=jnp.float32) for _ in range(num_layers)]
    init_c = [jnp.zeros((hidden_dim,), dtype=jnp.float32) for _ in range(num_layers)]
    xs = random.normal(keys[18], (seq_len, hidden_dim), dtype=jnp.float32)

    print("Executing backward pass for original 4-layer stacked LSTM...")
    grads = process_grad(init_h, init_c, xs)

    # Force evaluation
    grads.block_until_ready()
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark large function in scan loop"
    )
    parser.add_argument(
        "--unroll",
        type=int,
        default=1,
        help="Unroll factor for lax.scan (0 for full python unroll)",
    )
    args = parser.parse_args()

    run_benchmark(unroll=args.unroll)
