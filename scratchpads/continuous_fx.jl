using SymPy;
using LinearAlgebra;
# %%
@syms II,R,S,W, dt
# %%
V = diagm([II,II,II,II,II,II])
# %%
# Position Error update
V[1,2] = II*dt
# Velocity Error update
V[2,3] = -R*S*dt
V[2,4] = -R*dt
V[2,6] = II*dt
# Heading Error update
V[3,3] = II - W*dt
V[3,5] = -II*dt
# %%
V
# II think based on this design we only ever need to update 1,2 2,3 2,4 2,6
#
