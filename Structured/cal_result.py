import numpy as np
x_in = [  8.100875068441491, ]
x_out = [9.90053726286115, ] 

print("{:.2f} ± {:.2f}".format(np.mean(x_in), np.std(x_in)))
print("{:.2f} ± {:.2f}".format(np.mean(x_out), np.std(x_out)))