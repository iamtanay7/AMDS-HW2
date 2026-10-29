# msd_jax (core simulator)

The general mass-spring-damper simulator HW2 is built on. It is the same
physics as the course MATLAB fast solver. You do not need to modify it.

- `model.py`: model arrays (nodes, links, masses, fixed DOFs)
- `physics.py`: link forces and the semi-implicit Euler step
- `simulate.py`: general rollout with `jax.lax.scan`
