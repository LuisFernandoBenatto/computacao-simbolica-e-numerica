# In [1]: def erro_rel(p, pe):
# ...:     return abs(p - pe) / abs(p)
# ...:
#
# In [2]: def erro_real(p, pe):
# ...:     return p - pe
# ...:
#
# In [3]: def erro_abs(p, pe):
# ...:     return abs(p - pe)
# ...:
#
# In [4]: import numpy as np
#
# In [5]: np.pi
# Out[5]: 3.141592653589793
#
# In [6]: data = [('a', np.pi, 22/7), ('b', np.pi, 3.1416), ('d', np.sqrt(2), 1.414)]
#
# In [7]: data
# Out[7]:
# [('a', 3.141592653589793, 3.142857142857143),
# ('b', 3.141592653589793, 3.1416),
# ('d', 1.4142135623730951, 1.414)]
#
# In [8]: for item in data:
# ...:     letter, p, p_ = item
# ...:     relative_error = erro_real(p, p_)
# ...:     real_error = erro_real(p, p_)
# ...:     abs_error = erro_abs(p, p_)
# ...:     print(f"{letter} | erro relativo = {relative_error} | erro real = {real_error} | erro absoluto = {abs_error}")
# ...:
# a | erro relativo = -0.0012644892673496777 | erro real = -0.0012644892673496777 | erro absoluto = 0.0012644892673496777
# b | erro relativo = -7.346410206832132e-06 | erro real = -7.346410206832132e-06 | erro absoluto = 7.346410206832132e-06
# d | erro relativo = 0.00021356237309522186 | erro real = 0.00021356237309522186 | erro absoluto = 0.00021356237309522186
