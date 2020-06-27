
class AgentAdderConcatter:
    def __init__(self, agents, adder_fn):
        self._adders = {agent:adder_fn() for agent in agents}

    def get_example_output(self):
        example = next(iter(self._adders.values()))
        return example.get_example_output()

    def set_generate_callback(self, on_generate):
        for adder in self._adders.values():
            adder.set_generate_callback(on_generate)

    def add(self, agent, obs, action, reward, done, info):
        self._adders[agent].add(obs, action, reward, done, info)
