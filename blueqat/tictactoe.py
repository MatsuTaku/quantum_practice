from blueqat import opt
import numpy as np

opt.Opt().add('(q2-(q0+q1)/2-1/4)^2 + (q3-(q0+q2)/2+1/4)^2').add(np.diag([-1,-1,0,0])).run(shots=100)
