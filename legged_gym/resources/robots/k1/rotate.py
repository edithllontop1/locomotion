import numpy as np
from autolab_core import RigidTransform
R  = np.array([
    [  1.0000000,  0.0000000,  0.0000000,0],
   [0.0000000,  0.0007963,  -0.9999997,0],
   [0.0000000, 0.9999997,  0.0007963,0 ],
   [0,0,0,1]
])
rot = RigidTransform(translation=R[:3,3],rotation = R[:3,:3],from_frame='w',to_frame='r')
vec = RigidTransform(translation=np.array([0.0695,0.033,0]),from_frame='o',to_frame='w')

new_vec = rot*vec

print(new_vec.translation)