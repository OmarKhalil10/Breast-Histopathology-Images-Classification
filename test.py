import h5py

with h5py.File('_model_.h5', 'r') as f:
    for name in f:
        print(name)