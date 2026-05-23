# =============================================================================
# Shared mpmath / PyCall initializers for special-function Taylor tables
# =============================================================================
if !isdefined(@__MODULE__,:_mp)
    const _mp=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_mpctx)
    const _mpctx=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_mpf)
    const _mpf=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_mpc)
    const _mpc=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_pyfloat)
    const _pyfloat=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_cosh)
    const _cosh=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_sinh)
    const _sinh=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_legenq)
    const _legenq=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_hyperu)
    const _hyperu=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_gamma)
    const _gamma=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_exp)
    const _exp=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_cos)
    const _cos=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_sin)
    const _sin=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_digamma)
    const _digamma=Ref{PyObject}()
end
if !isdefined(@__MODULE__,:_pi)
    const _pi=Ref{PyObject}()
end

function __init_mpmath_specials__()
    m=pyimport("mpmath")
    _mp[]=m
    _mpctx[]=m.mp
    _mpf[]=m.mpf
    _mpc[]=m.mpc
    _pyfloat[]=pybuiltin("float")
    _cosh[]=m.cosh
    _sinh[]=m.sinh
    _legenq[]=m.legenq
    _hyperu[]=m.hyperu
    _gamma[]=m.gamma
    _exp[]=m.exp
    _cos[]=m.cos
    _sin[]=m.sin
    _digamma[]=m.digamma
    _pi[]=m.pi
    return nothing
end

__init_mpmath_specials__()