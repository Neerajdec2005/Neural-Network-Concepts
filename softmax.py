import numpy as np

layer_outputs= [[1.4, -2.15, -4, 3],
                [-1.2, 2.3, 0.8, -0.5],
                [0.1, -1.2, 3.4, -0.7]]

exp=np.exp(layer_outputs)

print(exp)
print()

norm_base= np.sum(exp, axis=1, keepdims=True)

softmax= exp/ norm_base
print(softmax)
print()
print(np.sum(softmax, axis=1))

