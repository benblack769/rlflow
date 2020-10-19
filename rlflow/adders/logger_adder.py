import numpy as np


class LoggerAdder:
    def __init__(self):
        self.on_generate = None
        self.reward_total = 0
        self.env_len = 0

    def get_example_output(self):
        return ("", 0.0)

    def set_generate_callback(self, on_generate):
        assert self.on_generate is None, "set_generate_callback should only be called once"
        self.on_generate = on_generate

    def add(self, obs, action, rew, done, info, actor_info):
        assert self.on_generate is not None, "need to call set_generate_callback before add"
        self.reward_total += rew
        self.env_len += 1
        logger_update = 50
        if done or self.env_len%logger_update == logger_update-1:
            self.on_generate(("mean", "reward_total", self.reward_total))
            self.on_generate(("mean", "env_len", self.env_len))
            self.on_generate(("sum", "env_steps", self.env_len))
            self.reward_total = 0
            self.env_len = 0
