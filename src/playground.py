import numpy as np

x = [ 4,6,8,34,32,5,67,8,9,76,5,4,3,5,6,76,7,8,8,9,5,4,4,3,347,3,5,63,65,6,36,245,25,4,642,645,63,2,54,2,546,57,347,35,24,2,54,642] 

gamma = 2
beta = 2

x = np.array(x)
print(f"This is the shape of x: {x.shape}")
batch_mean = np.mean(x)

print(f"This is the batch_mean: {batch_mean} and this is the shape: {batch_mean.shape}")

batch_variance = np.var(x)

print(f"This is the batch variance: {batch_variance}")

normalized = (x - batch_mean) / np.sqrt(batch_variance + 1e-8)


print(f"This is the normalized: {normalized}")

#scaling and shfiting, 
scaled = gamma * normalized + beta
print(f"this is the scaled output: {scaled}")
