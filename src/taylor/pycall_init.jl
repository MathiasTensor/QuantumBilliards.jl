# =============================================================================
# Shared mpmath / PyCall initializers for special-function Taylor tables
# =============================================================================
const _mp=Ref{PyObject}()
const _mpctx=Ref{PyObject}()
const _mpf=Ref{PyObject}()
const _mpc=Ref{PyObject}()
const _pyfloat=Ref{PyObject}()
const _mp_cosh=Ref{PyObject}()
const _mp_sinh=Ref{PyObject}()
const _mp_legenq=Ref{PyObject}()
const _mp_hyperu=Ref{PyObject}()
const _mp_gamma=Ref{PyObject}()
const _mp_exp=Ref{PyObject}()
const _mp_cos=Ref{PyObject}()
const _mp_sin=Ref{PyObject}()
const _mp_digamma=Ref{PyObject}()
const _mp_pi=Ref{PyObject}()
const _mp_hyp1f1=Ref{PyObject}()

function __init_mpmath_specials__()
    m=pyimport("mpmath")
    _mp[]=m
    _mpctx[]=m.mp
    _mpf[]=m.mpf
    _mpc[]=m.mpc
    _pyfloat[]=pybuiltin("float")
    _mp_cosh[]=m.cosh
    _mp_sinh[]=m.sinh
    _mp_legenq[]=m.legenq
    _mp_hyperu[]=m.hyperu
    _mp_gamma[]=m.gamma
    _mp_exp[]=m.exp
    _mp_cos[]=m.cos
    _mp_sin[]=m.sin
    _mp_digamma[]=m.digamma
    _mp_pi[]=m.pi
    _mp_hyp1f1[]=m.hyp1f1
    return nothing
end

__init_mpmath_specials__()