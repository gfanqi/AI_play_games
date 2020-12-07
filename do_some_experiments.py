import numpy as np

# a = np.random.random(size=(20, 4))
a = np.zeros(shape=(20,4))
# action = np.random.randint(0, 2, size=(20, 4))
action = np.zeros(shape=(20,4))
print(action)
b = np.random.random(size=(20, 4))

# action = np.argmax(action).astype(np.int)
# print(action)
# a[range(20),[0]*20] = 3

num_repeats = np.sum(action,axis=-1)

a[action == 1] = 0.5 * b[action == 1]
# print(a[action == 1])
print(a)
# print(np.repeat([1, 2, 3, 4, 5], [1, 2, 1, 2, 1]))