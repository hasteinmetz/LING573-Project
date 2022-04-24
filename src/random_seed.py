'''
Create and set random seeds to be used by the Python Dataloader
'''

import numpy as np

def write_random_seed(no_epochs=100):
    '''Get a random array of numbers to use as random seeds'''
    random_arr = np.random.randint(low=0, high=10000, size=(no_epochs,))
    np.savetxt(f'random_seeds_{no_epochs}.txt', random_arr)
    return random_arr

def load_random_seed(no_epochs=100):
    '''Load a random array of numbers to use as random seeds.
    If no file exists then create a random array of numbers'''
    try:
        seedfile = load_random_seed(f'random_seeds_{no_epochs}.txt', 'r')
        return seedfile
    except ValueError(f"No 'random_seeds_{no_epochs}.txt' found. Creating new values"):
        random_arr = write_random_seed(no_epochs)
        return random_arr

if __name__ == '__main__':
    write_random_seed()