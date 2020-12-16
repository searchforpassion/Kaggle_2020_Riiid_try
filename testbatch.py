import math
n = 101
bs = 10
list_a = list(range(n))
for b_i in range(math.ceil(n/bs)):
    if b_i*bs+bs < n:
        print(list_a[b_i*bs:b_i*bs+bs])
    else:
        print(list_a[b_i*bs:])