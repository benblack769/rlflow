import collections
import numpy as np
from .segment_tree import SumSegmentTree, MinSegmentTree

class BaseScheme:
    def add(self, id):
        '''
        args:
          id: unique id of data that needs to be created or updated
        '''
    def sample_batch(self):
        '''
        returns:
        - id of sampled data
        '''
    def remove(self, id):
        '''
        removes the id from the data
        '''
    def update_priorities(self, idxs, priorities):
        '''priority: priority of data (only needed for selectors which use it, can be ignored)'''


LLNode = collections.namedtuple("LLNode",["prev","next","value"])
class LList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self,val):
        new_node = LLNode(None,self.head,val)
        self.head = new_node
        if self.tail is None:
            self.tail = new_node
        return new_node

    def empty(self):
        return self.head is None

    def pop(self):
        val = self.tail.value
        self.tail = self.tail.prev

    def remove(self,node):
        if self.tail is self.head:
            self.tail = None
            self.head = None
        else:
            if node is self.head:
                self.head = node.next
            else:
                node.prev.next = node.next
            if node is self.tail:
                self.tail = node.prev
            else:
                node.next.prev = node.prev

class FifoScheme:
    def __init__(self):
        self.queue = LList()
        self.nodes = {}
    def add(self, id):
        node = self.queue.add(id)
        self.nodes[id] = node
    def sample(self):
        val = self.queue.pop()
        return val
    def remove(self, id):
        if id in self.nodes:
            self.queue.remove(self.nodes[id])
            del self.nodes[id]

class UniformSampleScheme:
    def __init__(self, seed=None):
        self.buffer = []
        self.id_mapper = {}
        self.np_random = np.random.RandomState(seed)
    def add(self, id):
        idx = len(self.buffer)
        self.id_mapper[id] = idx
        self.buffer.append(id)
    def sample(self):
        return self.np_random.randint(0,len(self.buffer))
    def remove(self, id):
        self.buffer[self.id_mapper[id]] = self.buffer[-1]
        self.buffer.pop()

class PrioritizedSampleScheme:
    def __init__(self, max_size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super().__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self.idx_mapper = {}
        self._next_idx = 0
        self._max_priority = 1.0

    def add(self, id):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = id

        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes, env=env)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities ** self._alpha
        self._it_min[idxes] = priorities ** self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
