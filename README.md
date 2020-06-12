### RL Pipelines

RL Pipelines is a set of platform agnostic tools that handles the data flow of reinforcement learning in a performant and composable way.

It is designed after Deepmind ACME ([code](https://github.com/deepmind/acme), [paper](https://arxiv.org/abs/2006.00979)), but does not depend on tensorflow or any other machine learning framework.

Features

* Vector environments to parallelize environment stepping
* Adders to transform step-wise data into a trainable format (built in options include normal transitions, n-step transitions, and sequences)
* Sample strategies for the learner to train on. Options include a first-in-first out pipeline, uniform replay buffer, prioritized replay buffer.
* Multiprocessing and multiprocessing agent utilities
* Environment loop to handle all this by

This data flow looks roughly like this

![data flow](diagrams/dataflow.svg)

### Learner class

The main job for the user is to define the learner.

The learner class needs to define a `step` method

Here is an example:

```
class Learner:
  def step(self, transition):
    pass
```

### Adder class

The adder class takes in states (observation, reward, done, info), and generates transitions, which are what the learner want.

Here is the most simple adder class, TransitionAdder:

```
class TransitionAdder:
    def __init__(self, on_generate):
        self.last_observation = None
        self.on_generate = on_generate

    def add(self, obs, rew, done, info):
        if self.last_observation is None:
            self.last_observation = obs
        else:
            if done:
                transition = (None, rew, self.last_observation)
            else:
                transition = (obs, rew, self.last_observation)
            self.on_generate(transition)
            self.last_observation = obs
```



### Selector class

The replay sampler is in charge of sampling data for training. For example prioritized experience replay is implemented by a specialized selector.

Here is what the selector class looks like:

```
class BaseScheme:
    def add(self, id, priority):
        '''
        args:
          id: unique id of data that needs to be created or updated
          priority: priority of data (only needed for selectors which use it, can be ignored)
        '''
    def sample(self):
        '''
        returns:
        - id of sampled data
        '''
    def remove(self, id):
        '''
        removes the id from the data
        '''
    def update_priorities(self, id, priority):
        '''
        Called by the learner.
        optional: only needed for prioritized experience replay
        '''
```

### Actor
