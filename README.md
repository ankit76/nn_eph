# Neural network wave functions for electron phonon coupled systems

Work in progress!

Requirements
------------

* python (>= 3.7)
* mpi4py
* numpy, scipy
* jax, jaxlib 
* flax

Examples
-------------

Add nn_eph to your PYTHONPATH and run the examples as:

```
  mpirun python holstein_1d.py > holstein_1d.out
  mpirun python holstein_2d.py > holstein_2d.out
```

Note that all sampling parameters are defined per process, so stochastic error should decrease if you use more processes for the same parameters.
