# In [33]: def erro_rel(p, pe):
# ...:     return abs(p - pe) / abs(p)
# ...:
#
# In [34]: def erro_real(p, pe):
# ...:     return p - pe
# ...:
#
# In [35]: def erro_abs(p, pe):
# ...:     return abs(p - pe)
# ...:
#
# In [36]: data2 = [('a', np.e**10, 22.000), ('d', np.math.factorial(9), np.sqrt(18*np.pi) * (9/np.e)**9)]
#
# In [37]: data2
# Out[37]: [('a', 22026.465794806703, 22.0), ('d', 362880, 359536.87284194835)]
#
# In [38]: np.e
# Out[38]: 2.718281828459045
#
# In [39]: np.math.factorial(9)
# Out[39]: 362880
#
# In [40]: for item in data2:
# ...:     letter, p, p_ = item
# ...:     relative_error = erro_rel(p, p_)
# ...:     real_error = erro_real(p, p_)
# ...:     abs_error = erro_abs(p, p_)
# ...:     print(f"{letter} | erro relativo = {relative_error} | erro real = {real_error} | erro absoluto = {abs_error}")
# ...:
# a | erro relativo = 0.9990012015452253 | erro real = 22004.465794806703 | erro absoluto = 22004.465794806703
# d | erro relativo = 0.009212762230080598 | erro real = 3343.1271580516477 | erro absoluto = 3343.1271580516477
