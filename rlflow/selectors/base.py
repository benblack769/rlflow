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
    def update_priorities(self, ids, priorities):
        '''priority: priority of data (only needed for selectors which use it, can be ignored)'''
