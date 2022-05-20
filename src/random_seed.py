'''
Create and set random seeds to be used by the Python Dataloader
'''

import numpy as np

def write_random_seed(filename, no_epochs=100):
    '''Get a random array of numbers to use as random seeds'''
    random_arr = np.random.randint(low=0, high=10000, size=(no_epochs+1,))
    np.savetxt(f'random_seeds_{no_epochs}.txt', random_arr)
    return random_arr

def load_random_seed(filename, no_epochs=100):
    '''Load a random array of numbers to use as random seeds.
    If no file exists then create a random array of numbers'''
    try:
        seedfile = open(filename, 'r')
        seeds = [int(float(x.strip())) for x in seedfile.readlines()]
        seedfile.close()
        return seeds
    except Exception:
        print(f"{filename} not found. Creating the file: random_seeds_{no_epochs}.txt")
        random_arr = write_random_seed(filename, no_epochs)
        return random_arr

if __name__ == '__main__':
    load_random_seed('./data/random_seeds_100.txt')