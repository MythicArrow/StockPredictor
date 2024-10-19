import jax
import jax.numpy as jnp
import optax

class Agent:
    def __init__(self, model, params, optimizer):
        self.model = model
        self.params = params
        self.optimizer = optimizer
        self.opt_state = optimizer.init(params)

    def predict(self, state):
        return self.model.apply(self.params, state)

    def update(self, batch_states, batch_actions, batch_targets):
        def loss_fn(params):
            q_values = self.model.apply(params, batch_states)
            q_value_pred = jax.vmap(lambda q, a: q[a])(q_values, batch_actions)
            return jnp.mean((batch_targets - q_value_pred) ** 2)

        grads = jax.grad(loss_fn)(self.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
