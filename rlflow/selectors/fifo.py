import collections
import numpy as np


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
    def sample(self, batch_size):
        if len(self.queue) >= batch_size:
            vals = [self.queue.pop() for i in range(batch_size)]
            return vals
        else:
            return None
    def remove(self, ids):
        for id in ids:
            if id in self.nodes:
                self.queue.remove(self.nodes[id])
                del self.nodes[id]
