# =============================================================================
# Shared mpmath / PyCall initializers for special-function Taylor tables
# =============================================================================

# reentrant lock for the mpamth seeding, does not like to play with Julia's GC someties.
const PYCALL_MPMATH_LOCK=ReentrantLock()

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
const _mp_polygamma=Ref{PyObject}()
const _mp_pi=Ref{PyObject}()
const _mp_hyp1f1=Ref{PyObject}()
const _mp_legenp=Ref{PyObject}()
const _mp_airyai=Ref{PyObject}()
const _mp_airybi=Ref{PyObject}()

const _py_seed_q=PyNULL()
const _py_seed_p=PyNULL()
const _py_seed_u=PyNULL()
const _py_seed_a=PyNULL()
const _py_mag_consts=PyNULL()

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
    _mp_polygamma[]=m.polygamma
    _mp_pi[]=m.pi
    _mp_hyp1f1[]=m.hyp1f1
    _mp_legenp[]=m.legenp
    _mp_airyai[]=m.airyai
    _mp_airybi[]=m.airybi
    return nothing
end

function __init_legendre_seed_helpers__()
    py"""
    import mpmath as mp

    def qb_seed_q(nu_re,nu_im,d0,dps,leg_type):
        mp.mp.dps=int(dps)
        nu=mp.mpc(nu_re,nu_im)
        d=mp.mpf(d0)
        z=mp.cosh(d)
        sh=mp.sinh(d)
        Q0=mp.legenq(nu,0,z,type=int(leg_type))
        Q1=mp.legenq(nu+1,0,z,type=int(leg_type))
        y=(nu+1)*(Q1-z*Q0)/sh
        return float(mp.re(Q0)),float(mp.im(Q0)),float(mp.re(y)),float(mp.im(y))

    def qb_seed_p(nu_re,nu_im,d0,dps):
        mp.mp.dps=int(dps)
        nu=mp.mpc(nu_re,nu_im)
        d=mp.mpf(d0)
        z=mp.cosh(d)
        sh=mp.sinh(d)
        P0=mp.legenp(nu,0,z)
        P1=mp.legenp(nu+1,0,z)
        y=(nu+1)*(P1-z*P0)/sh
        return float(mp.re(P0)),float(mp.im(P0)),float(mp.re(y)),float(mp.im(y))
    """
    copy!(_py_seed_q,py"qb_seed_q")
    copy!(_py_seed_p,py"qb_seed_p")
    return nothing
end

function __init_confluent_u_helpers__()
    py"""
    import mpmath as mp
    def qb_seed_u(nu_re,nu_im,s0,dps):
        mp.mp.dps=int(dps)
        nu=mp.mpc(nu_re,nu_im)
        s=mp.mpf(s0)
        z=s*s
        a=mp.mpf('0.5')-nu
        C=-mp.mpf('0.25')/mp.gamma(nu+mp.mpf('0.5'))
        U0=mp.hyperu(a,1,z)
        U1=mp.hyperu(a+1,2,z)
        ez=mp.exp(-z/2)
        F=C*ez*U0
        Fz=C*ez*(-mp.mpf('0.5')*U0-a*U1)
        G=F
        Gp=2*s*Fz
        return (float(mp.re(G)),float(mp.im(G)),float(mp.re(Gp)),float(mp.im(Gp)))

    def qb_seed_a(nu_re,nu_im,s0,dps):
        mp.mp.dps=int(dps)
        nu=mp.mpc(nu_re,nu_im)
        s=mp.mpf(s0)
        z=s*s
        a=mp.mpf('0.5')-nu
        c=mp.cos(mp.pi*nu)/(4*mp.pi)
        M0=mp.hyp1f1(a,1,z)
        M1=mp.hyp1f1(a+1,2,z)
        ez=mp.exp(-z/2)
        A=c*ez*M0
        Az=c*ez*(a*M1-mp.mpf('0.5')*M0)
        Ap=2*s*Az
        return (float(mp.re(A)),float(mp.im(A)),float(mp.re(Ap)),float(mp.im(Ap)))

    def qb_mag_consts(nu_re,nu_im,dps):
        mp.mp.dps=int(dps)
        nu=mp.mpc(nu_re,nu_im)
        A0=mp.cos(mp.pi*nu)/(4*mp.pi)
        S0=mp.sin(mp.pi*nu)
        D=mp.digamma(nu+mp.mpf('0.5'))-2*mp.digamma(mp.mpf('1'))
        P1=mp.polygamma(1,nu+mp.mpf('0.5'))
        P2=mp.polygamma(2,nu+mp.mpf('0.5'))
        R0=A0*D-S0/4
        A1=-S0/4
        A2=-mp.pi*mp.cos(mp.pi*nu)/4
        R1=A1*D+A0*P1-mp.pi*mp.cos(mp.pi*nu)/4
        R2=A2*D+2*A1*P1+A0*P2+(mp.pi**2)*S0/4
        return (float(mp.re(A0)),float(mp.im(A0)),float(mp.re(R0)),float(mp.im(R0)),float(mp.re(R1)),float(mp.im(R1)),float(mp.re(R2)),float(mp.im(R2)))
    """
    copy!(_py_seed_u,py"qb_seed_u")
    copy!(_py_seed_a,py"qb_seed_a")
    copy!(_py_mag_consts,py"qb_mag_consts")
    return nothing
end

__init_mpmath_specials__()
__init_legendre_seed_helpers__()
__init_confluent_u_helpers__()