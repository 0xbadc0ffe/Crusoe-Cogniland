import jax


class RNGManager:
    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        self._stack = []

    def get_key(self):
        self._key, sub = jax.random.split(self._key)
        return sub

    def checkpoint(self):
        self._stack.append(self._key)

    def restore(self):
        self._key = self._stack.pop()
