import collections
import numpy as np
from .base import BaseScheme

class LLNode:
    def __init__(self, prev, next, val):
        self.prev = prev
        self.next = next
        self.value = val

class LList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def add(self,val):
        old_head = self.head
        new_node = LLNode(None,old_head,val)
        self.head = new_node
        if old_head is not None:
            old_head.prev = new_node
        else:
            self.tail = new_node
        self.size += 1
        return new_node

    def empty(self):
        return self.head is None

    def pop(self):
        val = self.tail.value
        self.remove(self.tail)
        return val

    def remove(self,node):
        if self.head is not None:
            self.size -= 1

        if self.tail is self.head:
            self.tail = None
            self.head = None
        else:
            if node is self.head:
                self.head = node.next
                self.head.prev = None
            else:
                node.prev.next = node.next

            if node is self.tail:
                self.tail = node.prev
                self.tail.next = None
            else:
                node.next.prev = node.prev

    def __len__(self):
        return self.size

class FifoScheme(BaseScheme):
    def __init__(self):
        self.queue = LList()
        self.nodes = {}
    def add(self, id):
        node = self.queue.add(id)
        self.nodes[id] = node
    def sample(self, batch_size):
        if len(self.queue) >= batch_size:
            vals = [self.queue.pop() for i in range(batch_size)]
            for val in vals:
                del self.nodes[val]
            return vals
        else:
            return None
    def remove(self, id):
        if id in self.nodes:
            self.queue.remove(self.nodes[id])
            del self.nodes[id]
