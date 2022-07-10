# Geometry-Aware Dexterous Manipulation in PyBullet
## classification
- data for training is limited, evaluation is not well
## train_pointnet
- point cloud of object above palm has impact on training
- rotation difference between object and target is not accuracy
- capture point cloud every step is too slow
- capture point cloud of target, get point cloud 1 by rotation 1, get point cloud 2 by rotation 2, calculate rotation difference
