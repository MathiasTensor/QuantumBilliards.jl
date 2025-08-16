"""
    momentum_representation_of_wavefunction(us::Vector{T}, pts::BP, k::T) -> Function
        where {T<:Real, BP<:AbsPoints}

Builds a callable that evaluates the momentum-space wavefunction ψ̂(p) from boundary data.

Inputs
- us::Vector{T}: Normal derivative of the Dirichlet eigenfunction on the full boundary,
  sampled at quadrature points ordered by arclength.
- pts::BP: Boundary points container with:
    • pts.xy::Vector{SVector{2,T}}  — Cartesian boundary points q_j = (x_j, y_j)  
    • pts.ds::Vector{T}             — arclength weights Δs_j for each q_j
- k::T: Wavenumber of the state (the classical energy shell is |p| = k).

Return
- A function `mom(p::SVector{2,T})::Complex{T}` that evaluates ψ̂(p).

Details
- Off the energy shell (|p|^2 ≠ k^2): uses Bäcker Eq. (11)
    ψ̂(p) = (1 / 2π) * I(p) / (|p|^2 − k^2),
    where I(p) ≈ Σ_j u_j * exp(−i p⋅q_j) * Δs_j.
- Near the shell (|p| ≈ k): uses Bäcker Eq. (13)
    ψ̂(p) = −i / (4π k^2) * ∫ exp(−i p⋅q) * (p⋅q) * u(q) ds
          ≈ −i / (4π k^2) * Σ_j (p⋅q_j) u_j exp(−i p⋅q_j) Δs_j.
- The “near shell” switch uses tolerance √eps(T).

Performance
- One evaluation is O(N) with N = length(pts.xy).
- Precomputes x_j, y_j and w_j = u_j * Δs_j to avoid allocations; inner loops use @inbounds/@simd.

Assumptions
- Boundary data must already include any symmetry reflections (i.e., defined on the full boundary).
- ψ̂(p) is complex in general; for time-reversal invariant systems one may choose real position-space eigenfunctions but ψ̂ remains complex.

Reference
- A. Bäcker, “Momentum representation of eigenfunctions,” Eq. (11) and Eq. (13).
"""
function momentum_representation_of_wavefunction(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real,BP<:AbsPoints}
    N=length(pts.xy)
    x=Vector{T}(undef,N)
    y=Vector{T}(undef,N)
    w=Vector{T}(undef,N) # w_j = u_j * ds_j
    @inbounds for j in 1:N
        x[j]=pts.xy[j][1]
        y[j]=pts.xy[j][2]
        w[j]=us[j]*pts.ds[j]
    end
    k2=k*k
    thr=sqrt(eps(T))
    inv2π=inv(T(2π))
    I_T=complex(zero(T),one(T)) # 0 + 1im of type T
    function mom(p::SVector{2,T})
        px,py=p
        # I(p) = ∫ u(s) e^{-i p·q} ds  ≈  Σ w_j cis(-p·q_j)
        accI=zero(Complex{T})
        @inbounds @simd for j in 1:N
            accI+=w[j]*cis(-(muladd(px,x[j],py*y[j])))
        end
        p2=muladd(px,px,py*py)
        δ=p2-k2
        if abs(δ)>thr # off shell (Bäcker Eq. 11)
            return inv2π*accI/δ
        else  # on shell (Eq. 13)
            accOn=zero(Complex{T})
            @inbounds @simd for j in 1:N
                pq=muladd(px, x[j], py*y[j])
                accOn+=us[j]*pts.ds[j]*cis(-pq)*pq
            end
            return -(I_T/(T(4π)*k2))*accOn
        end
    end
    return mom
end
    
"""
    computeRadiallyIntegratedDensity(us::Vector{T}, pts::BP, k::T) -> Function
        where {T<:Real, BP<:AbsPoints}

Builds a callable that evaluates the radially integrated momentum density
I(φ) = ∫₀^∞ |ψ̂(r p̂_φ)|² r dr, where p̂_φ = (cos φ, sin φ).

Inputs
- us::Vector{T}: Normal derivative on the full boundary at the quadrature points.
- pts::BP: Boundary points with:
    • pts.xy::Vector{SVector{2,T}}  — boundary coordinates q_i  
    • pts.ds::Vector{T}             — arclength weights Δs_i
- k::T: Wavenumber.

Return
- A function `I_phi(φ::T)::T` that returns the angular distribution I(φ) (real, non-negative up to numerical noise).

Formula used (Bäcker Eqs. 24–25)
- Let α_ij(φ) = | p̂_φ ⋅ (q_i − q_j) | and x_ij = k * α_ij.
- Kernel: f(x) = sin(x) * Ci(x) − cos(x) * Si(x), where Ci and Si are cosine/sine integrals.
- I(φ) = (1 / 8π²) * Σ_{i,j} u_i u_j Δs_i Δs_j f( x_ij )  −  (∫ u ds)² / (2 k²).
  The last term vanishes for odd symmetry classes where ∫ u ds = 0, but we still calculate it.

Numerics
- Uses upper-triangular summation with symmetry factor 2 to halve work; the diagonal is added once.
- Clamps arguments to special functions away from zero via max(|x|, √eps(T)).
- Complexity per evaluation: O(N²).

Reference
- A. Bäcker, “Momentum representation of eigenfunctions,” Eqs. (24)–(25).
"""
function computeRadiallyIntegratedDensity(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real, BP<:AbsPoints}
    N=length(pts.xy)
    x=Vector{T}(undef,N)
    y=Vector{T}(undef,N)
    w=Vector{T}(undef,N) # w_i = u_i * ds_i
    @inbounds for i in 1:N
        x[i]=pts.xy[i][1]
        y[i]=pts.xy[i][2]
        w[i]=us[i]*pts.ds[i]
    end
    # constant term (∫u ds)^2/(2k^2)
    s0=sum(w)
    const_term=(s0*s0)/(T(2)*k*k)
    thr=sqrt(eps(T))
    inv8π2=inv(T(8π^2))
    function I_phi(φ::T)
        c,s=cos(φ),sin(φ)
        acc=zero(T)
        # use upper triangle + symmetry factor 2 to halve work
        @inbounds for i in 1:N
            xi,yi,wi=x[i],y[i],w[i]
            # diagonal (i==i) contributes once
            αii=abs(c*0+s*0)
            xii=max(abs(αii*k),thr)
            acc+=wi*wi*(sin(xii)*cosint(xii)-cos(xii)*sinint(xii))
            for j in (i+1):N
                α=abs(muladd(c,xi-x[j],s*(yi-y[j])))
                z=max(abs(α*k),thr)
                f=sin(z)*cosint(z) - cos(z)*sinint(z)
                acc+=T(2)*wi*w[j]*f  # symmetry (i,j) + (j,i)
            end
        end
        return inv8π2*acc-const_term
    end
    return I_phi
end
    
"""
    computeAngularIntegratedMomentumDensity(us::Vector{T}, pts::BP, k::T) -> Function
        where {T<:Real, BP<:AbsPoints}

Builds a callable that evaluates the angularly integrated momentum density
R(r) = ∫₀^{2π} |ψ̂(p̂_r p̂_φ)|² dφ.

Inputs
- us::Vector{T}: Normal derivative on the full boundary at quadrature points.
- pts::BP: Boundary points with:
    • pts.xy::Vector{SVector{2,T}}  — boundary coordinates q_i  
    • pts.ds::Vector{T}             — arclength weights Δs_i
- k::T: Wavenumber.

Return
- A function `R_r(r::T)::T` that returns R(r) at radius (wavenumber) r ≥ 0.

Formulas (Bäcker)
- Generic r (Eq. 28):
    R(r) = (1 / 2π) * [ r / (r² − k²)² ] * Σ_{i,j} u_i u_j Δs_i Δs_j J₀( r * |q_i − q_j| ).
- Near the energy shell r ≈ k (Eq. 31, shell approximation):
    R(r) = (1 / 16π k) * Σ_{i,j} u_i u_j Δs_i Δs_j * (|q_i − q_j|² / 2) * [ J₂(r d_ij) − J₀(r d_ij) ],
    where d_ij = |q_i − q_j| and J₀, J₂ are Bessel J of orders 0 and 2.

Numerics
- Switches to the on-shell expression when |r − k| < √eps(T).
- Uses upper-triangular summation with symmetry factor 2; diagonal handled separately.
- Complexity per evaluation: O(N²). Uses `Bessels.besselj`.

Notes
- R(r) should be non-negative up to numerical round-off; small negative dips near the shell can occur due to cancellation.
- Boundary data must cover the full boundary (after symmetry application).

Reference
- A. Bäcker, “Momentum representation of eigenfunctions,” Eqs. (28) and (31).
"""
function computeAngularIntegratedMomentumDensity(us::Vector{T},pts::BP,k::T) :: Function where {T<:Real,BP<:AbsPoints}
    N=length(pts.xy)
    x=Vector{T}(undef, N)
    y=Vector{T}(undef, N)
    w=Vector{T}(undef, N) # w_i = u_i * ds_i
    @inbounds for i in 1:N
        x[i]=pts.xy[i][1]
        y[i]=pts.xy[i][2]
        w[i]=us[i]*pts.ds[i]
    end
    k2=k*k
    thr=sqrt(eps(T))
    inv2π=inv(T(2π))
    function R_r(r::T)
        if abs(r-k)<thr # near shell (Eq. 31)
            acc=zero(T)
            @inbounds for i in 1:N
                xi,yi,wi=x[i],y[i],w[i]
                # diagonal
                z=zero(T)
                J0=Bessels.besselj(0,z) # = 1
                J2=Bessels.besselj(2,z) # = 0
                acc+=wi*wi*(zero(T)^2*T(0.5)*(J2-J0)) # = 0
                for j in (i+1):N
                    dx=xi-x[j];dy=yi-y[j]
                    d=hypot(dx, dy)
                    z=d*r
                    J0=Bessels.besselj(0, z)
                    J2=Bessels.besselj(2, z)
                    acc+=T(2)*wi*w[j]*(d^2*T(0.5)*(J2-J0))
                end
            end
            return acc/(T(16π)*k)
        else # general r (Eq. 28)
            acc=zero(T)
            @inbounds for i in 1:N
                xi,yi,wi=x[i],y[i],w[i]
                acc+=wi*wi*Bessels.besselj(0,zero(T)) # = wi^2 * 1
                for j in (i+1):N
                    d=hypot(xi-x[j],yi-y[j])
                    acc+=T(2)*wi*w[j]*Bessels.besselj(0,d*r)
                end
            end
            return inv2π*(r/((r*r-k2)^2))*acc
        end
    end
    return R_r
end
