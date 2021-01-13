import numpy as np
np.set_printoptions(suppress=True, precision=3)

#this code contains some tests i did with cosine similarity

def cosine_similarity(x):
    x_normed = x/np.linalg.norm(x,axis=-1,keepdims=True)
    result = np.einsum("ab,cb->ac", x_normed, x_normed)
    return result
    
#set up values
indices = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5])
values = np.random.uniform(size=[6,768],low=-1.0,high=1.0)
values = values[indices]

#calculate direct equality
n_values = values.shape[0]
equals = np.zeros([n_values,n_values],dtype=np.int32)
for i in range(n_values):
    for j in range(n_values):
        equals[i,j] = np.all(values[i]==values[j])
equals_sum = np.sum(equals,axis=-1)

#calculate with cosine similarity
values_cs = cosine_similarity(values)
values_cs = np.fmax(values_cs, np.zeros_like(values_cs)) #ReLU
values_sum = np.sum(values_cs,axis=-1)

print("Result should be:")
print(equals_sum) 
print("Result is:")
print(values_sum)
