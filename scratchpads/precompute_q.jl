using SymPy;
using LinearAlgebra;
# %%
@syms an wn aw ww dt
# %%
Qi = diagm([an * dt^2, wn * dt^2, aw * dt, ww * dt, 0])
# %%
Fi = diagm(6,5,-1 => [1,1,1,1,1])
# %%
Fi * Qi * Fi'
# go from 3 to 12
# aka g from 1 to 4
