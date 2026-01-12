"""
Contains a pivoted-Cholesky based numerical-range extraction and a stable solver
for the symmetric generalized eigenvalue problem

    B x = μ A x     (equivalently A x = λ B x with μ = 1/λ)

when `A` is positive semidefinite but numerically singular.

----------------------------------------------------------------------
1) Pivoted Cholesky: numerical rank and range of A
----------------------------------------------------------------------

`_cholpiv_factor(A; eps_rank)` computes a pivoted Cholesky factorization

    P' A P ≈ U' U

where `P` is a permutation matrix and `U` is upper-triangular.
The factorization is stopped when the remaining pivots fall below `eps_rank`,
so the number of nonzero pivots

    r0 = Fchol.rank

is the numerical rank of `A`, i.e. the dimension of its numerical range

    range(A) = span{ eigenvectors with eigenvalue > eps }.

This gives the size of the physically meaningful subspace.

----------------------------------------------------------------------
2) Initial basis for range(A)
----------------------------------------------------------------------

`_cholpiv_fill!` builds a basis `J ∈ ℝ^{n×r0}` for `range(A)` from the Cholesky
factor. Internally it inverts the leading block of `U` to obtain vectors
spanning the same range as `A`.

We then add `pad` random columns to avoid missing important directions:

    J ← [ J   Ω ],     Ω ∈ ℝ^{n×pad}

so that `J ∈ ℝ^{n×m}` with `m = r0 + pad`.

Finally we orthonormalize:

    J ← orth(J)

so `J'J ≈ I`.

This gives a *rough* basis for the numerical range of `A`.

----------------------------------------------------------------------
3) Subspace improvement:  J ← orth(A J)
----------------------------------------------------------------------

We perform one “power-type” subspace sweep

    AJ = A*J
    J  = orth(AJ)
    AJ = A*J

Mathematically, if

    A = S diag(d) S',    d₁ ≥ d₂ ≥ ⋯ ≥ 0,

then multiplying by `A` scales each component by `dᵢ`.
Directions belonging to the numerical nullspace (`dᵢ ≈ 0`) are suppressed,
while true range directions are amplified.

After orthonormalization this produces a much cleaner basis for `range(A)`.
The final `AJ = A*J` is cached for later.

----------------------------------------------------------------------
4) Projection to the reduced problem
----------------------------------------------------------------------

We project the operators to the subspace:

    Ared = J' * A * J   = J' * AJ
    Bred = J' * B * J

These are small (`m×m`) symmetric matrices representing the generalized
eigenproblem restricted to the numerical range of `A`.

----------------------------------------------------------------------
5) Removal of the remaining nullspace inside Ared
----------------------------------------------------------------------

We diagonalize

    Ared = U diag(d) U'

and keep only eigenvalues

    d > max(eps_rank * d_max, eps(T)).

This removes any nullspace that survived the projection step.
Ukeep, dkeep are the kept eigenvectors and eigenvalues.

----------------------------------------------------------------------
6) Whitening (conversion to a standard eigenproblem)
----------------------------------------------------------------------

We build

    T = Ukeep * diag(1 / sqrt(dkeep)),

which satisfies

    T' * Ared * T = I.

Then

    B2 = T' * Bred * T

is the reduced operator in a basis where `A = I`.
The generalized problem

    Bred y = μ Ared y

has become the standard symmetric problem

    B2 z = μ z.

----------------------------------------------------------------------
7) Back-transformation to the physical space
----------------------------------------------------------------------

The eigenvectors of the original problem are

    x = J * T * z.

The matrix

    S2 = J * T

is therefore a whitened basis for the numerical range of `A`
satisfying

    S2' * A * S2 = I.

When `want_vecs = true`, the function returns `(μ, Z, S2)`,
so that physical coefficient vectors are obtained as

    X = S2 * Z.

MO/12/1/26
"""

"""
    _thin_qr!(Y::Matrix{T}) where {T<:Real} -> Matrix{T}

In-place thin QR that overwrites `Y` with the thin Q factor.

Used to re-orthonormalize a tall matrix without forming `R` explicitly.

# Inputs
- `Y::Matrix{T}` (size `n×m`, `n ≥ m` recommended):
  On entry, a dense matrix whose columns span some subspace.
  On exit, `Y` is overwritten with `Q` (still stored as an `n×m` dense matrix).

# Outputs
- Returns the same matrix `Y`, overwritten with `Q`.

# Logic
1. Calls `LAPACK.geqrf!(Y)` to compute a QR factorization in-place:
   the upper triangle of `Y` holds `R`, and the Householder reflectors are stored
   in the strictly-lower triangle plus the vector `τ`.
2. Calls `LAPACK.orgqr!(Y, τ, m)` to explicitly form the first `m` columns of `Q`
   and overwrite `Y` with that `Q`.
"""
@inline function _thin_qr!(Y::Matrix{T}) where {T<:Real}
    Y,τ=LAPACK.geqrf!(Y)
    LAPACK.orgqr!(Y,τ,size(Y,2))
    return Y
end

"""
    _sym_trans_X_Y!(G::StridedMatrix{T}, X::StridedMatrix{T}, Y::StridedMatrix{T}) where {T} -> StridedMatrix{T}

Compute `G = X' * Y` in-place. This is the core for forming reduced matrices
`Ared = J'*(A*J)` and `Bred = J'*(B*J)`), where `J` is a subspace basis.

# Inputs
- `G::StridedMatrix{T}` (size `m×m`):
  Destination matrix overwritten with `X'Y`.
- `X::StridedMatrix{T}` (size `n×m`):
  Left factor.
- `Y::StridedMatrix{T}` (size `n×m`):
  Right factor.

# Outputs
- Returns `G` (overwritten).
"""
@inline function _sym_trans_X_Y!(G::StridedMatrix{T},X::StridedMatrix{T},Y::StridedMatrix{T}) where {T}
    mul!(G,adjoint(X),Y)
    return G
end

"""
    _cholpiv_factor(A::Symmetric{T,Matrix{T}}; eps_rank::T = T(1e-15)) where {T<:Real}
        -> (F, tol)

Compute a **pivoted Cholesky factorization** of a (numerically) positive semidefinite
symmetric matrix `A` and return a rank-revealing factor object `F` plus the effective tolerance.

This is used to cheaply estimate the numerical rank and to build a stable
approximate basis for `range(A)` (equivalently the complement of the numerical nullspace),
at the same tolerance scale used elsewhere.

# Inputs
- `A::Symmetric{T,Matrix{T}}`:
  Real symmetric matrix. In your workflow this is typically the Grammian-like matrix
  arising from Vergini–Saraceno / scaling constructions, expected to be PSD up to
  numerical error.
- `eps_rank::T` (keyword, default `1e-14`):
  Relative tolerance multiplier used to decide when to stop pivoting / what rank to keep.
  The effective absolute tolerance is computed from the diagonal scale:
  `tol = max(eps_rank * maximum(abs, diag(A)), eps(T))`.

# Outputs
- `F`: The factorization object returned by
  `cholesky(Hermitian(parent(A)), Val(true); tol=tol, check=false)`.
  Important fields used later:
  - `F.rank :: Int`  : estimated numerical rank at the given `tol`
  - `F.p    :: Vector{Int}` : pivot permutation
  - `F.U    :: Matrix{T}`   : upper-triangular factor in pivoted ordering (packed)
- `tol::T`: The absolute tolerance actually passed to the factorization.

# Caveats
- If `A` is strongly indefinite, pivoted Cholesky can fail -> RRQR/SVD-type rank revealing TODO.
"""
function _cholpiv_factor(A::Symmetric{T,Matrix{T}};eps_rank::T=T(1e-15)) where {T<:Real}
    Ah=Hermitian(parent(A))
    maxdiag=maximum(abs,diag(Ah))
    tol=max(eps_rank*maxdiag,eps(T))
    F=cholesky(Ah,Val(true);tol=tol,check=false)
    return F,tol
end

"""
    _cholpiv_fill!(J::Matrix{T}, F; r0::Int = F.rank) where {T<:Real} -> Matrix{T}

Fill the first `r0` columns of `J` with a range(A) basis implied by the pivoted-Cholesky factor `F`.

This is the basis builder: it constructs a matrix whose columns span the
numerical range of `A` (at the pivot tolerance), without explicitly forming eigenvectors.

In your pipeline it is used as:
- compute `F = cholesky(Hermitian(A), Val(true); tol=..., check=false)`
- set `r0 = F.rank`
- allocate `J` with `m ≥ r0` columns
- call `_cholpiv_fill!(J, F; r0=r0)`
- optionally append random padding in columns `r0+1:m`
- orthonormalize via thin QR, then apply one sweep `J ← orth(A*J)`.

# Inputs
- `J::Matrix{T}` (size `n×m`):
  Destination matrix. Only columns `1:r0` are written; the rest are left unchanged.
  Must satisfy `r0 ≤ m`.
- `F`:
  The pivoted-Cholesky factorization object returned by `_cholpiv_factor`.
  Must provide fields:
  - `F.rank`, `F.p`, and `F.U`.
- `r0::Int` (keyword, default `F.rank`):
  Number of columns/basis vectors to construct (the numerical rank).

# Output
- Returns `J` with `J[:,1:r0]` filled.

# Mathematical meaning 
For a pivoted Cholesky factorization of `P' A P ≈ U' U` with rank `r0`, the leading block
`U11 = U[1:r0,1:r0]` is upper triangular and well-conditioned relative to the tolerance.
A convenient basis for the effective range is given by

    J[P[1:r0], 1:r0] = inv(U11)

and zeros elsewhere. This spans the same subspace as the significant columns selected by pivoting.

# Implementation details
- Extract `U11 = view(F.U, 1:r0, 1:r0)` and permutation `p = F.p`.
- Form `X = I` (size `r0×r0`) and solve `U11 * X = I` in-place via `ldiv!`,
  thus `X = inv(U11)`.
- Zero out `J[:,1:r0]`.
- For each `i=1:r0`, write row `X[i,:]` into row `p[i]` of `J[:,1:r0]`.
"""
function _cholpiv_fill!(J::Matrix{T},F;r0::Int=F.rank) where {T<:Real}
    r0==0 && return J
    p=F.p;U11=@view F.U[1:r0,1:r0]
    X=Matrix{T}(I,r0,r0)
    ldiv!(UpperTriangular(U11),X) 
    fill!(@view(J[:,1:r0]),zero(T))
    @inbounds for i in 1:r0
        _pi=p[i]
        @views J[_pi,1:r0].=X[i,:]
    end
    return J
end

"""
    _improve_subspace!(J::Matrix{T}, AJ::Matrix{T}, A::Symmetric{T,Matrix{T}}) where {T<:Real} -> Matrix{T}

Perform a single subspace improvement step and return `AJ = A*J` for reuse in the reduced solve.

# Inputs
- `J::Matrix{T}` (size `n×m`):
  Current subspace basis (columns). On exit, overwritten with an improved basis.
- `AJ::Matrix{T}` (size `n×m`):
  Workspace to store `A*J`. Must be preallocated to match `J`.
  On exit, contains the final `AJ = A*J` corresponding to the updated `J`.
- `A::Symmetric{T,Matrix{T}}` (size `n×n`):
  Symmetric operator applied to the subspace.

# Output
- Returns `AJ`, containing `A*J` for the updated `J`.

#  Notes
- `J` has orthonormal columns approximating dominant range directions of `A`
- `AJ` is consistent with that `J` and can be fed directly into `Ared = J' * AJ` without recomputing `A*J` inside the reduced stage.
"""
function _improve_subspace!(J::Matrix{T},AJ::Matrix{T},A::Symmetric{T,Matrix{T}}) where {T<:Real}
    mul!(AJ,A,J) 
    _thin_qr!(AJ) 
    copyto!(J,AJ) 
    mul!(AJ,A,J) 
    return AJ
end

"""
    generalized_eig_cholesky(A::Symmetric{T,Matrix{T}}, B::Symmetric{T,Matrix{T}};eps_rank::T = T(1e-14),want_vecs::Bool=false) where {T<:Real} -> Vector{T}

Compute generalized eigenvalues `μ` of the pencil `(B, A)` restricted to the numerical range
of `A`, using a pivoted-Cholesky + one-sweep subspace. This is made from A x = λ B x where λ = 1/μ,
which is the original GEP.

# Steps
1) estimating the numerical rank/range of `A` via pivoted Cholesky,
2) building a trial subspace `J` of size `n×m` with `m ≈ r0 + pad`,
3) doing one subspace-improvement step `J ← orth(A*J)`,
4) forming reduced matrices `Ared = J'*(A*J)` and `Bred = J'*(B*J)`,
5) filtering out the numerical nullspace of `Ared`,
6) returning the eigenvalues of the final reduced SPD/indefinite pencil in standard form.

# Inputs
- `A::Symmetric{T,Matrix{T}}` (size `n×n`): Real symmetric matrix, expected to be positive semidefinite (PSD) up to roundoff.
  Its numerical nullspace is the key object: only directions with `A` above tolerance.
  are kept.
- `B::Symmetric{T,Matrix{T}}` (size `n×n`): Real symmetric matrix defining the generalized problem.
- `eps_rank::T` (keyword, default `1e-14`): Relative tolerance controlling (i) pivoted-Cholesky stopping and (ii) the thresholding
  of eigenvalues of `Ared`.
- `want_vecs::Bool` (keyword, default `false`): Whether to compute and return eigenvectors along with eigenvalues.

# Outputs
- `μ::Vector{T}`: Generalized eigenvalues (real) corresponding to the numerically non-null subspace.
  The length is the selected rank `r` (typically equal to the pivoted-Cholesky rank).

# High-level mathematics
We want generalized eigenpairs (in the well-conditioned subspace)
    B v = μ A v
but `A` is numerically singular. The standard safe approach is to restrict to the range of `A`
and “whiten” it:
- Find a basis `S2` spanning `range(A)` (numerically).
- Form `A2 = S2' A S2` and `B2 = S2' B S2`.
- In that basis, discard tiny eigenvalues of `A2`, and scale by `A2^{-1/2}`:
    T = Ukeep * diag(1/sqrt(dkeep))
    B̃ = T' B2 T
  Then `eigvals(B̃)` are the generalized eigenvalues.

This function implements exactly that, but uses a cheap subspace construction for `S2`.

# Algorithm (step-by-step)
Let `pad = 50` (hard-coded here; tune as needed).

1. **Pivoted Cholesky rank estimate**
   - `Fchol, _ = _cholpiv_factor(A; eps_rank=eps_rank)`
   - `r0 = Fchol.rank` is the estimated numerical rank of `A`.

2. **Build trial subspace J (n×m)**
   - `m = min(n, r0 + pad)`
   - Fill first `r0` columns from the pivoted factor: `_cholpiv_fill!(J, Fchol; r0=r0)`
   - Fill padding columns with noise: `J[:,r0+1:m] .= randn`
   - Orthonormalize: `_thin_qr!(J)` so `J'J ≈ I`.

3. **One-sweep subspace improvement**
   - `_improve_subspace!(J, AJ, A)` performs:
     - `AJ = A*J`
     - `AJ = orth(AJ)` (thin QR)
     - `J  = AJ`
     - `AJ = A*J` (final, consistent with updated `J`)
   After this, `J` is a better range basis, and `AJ` is cached for the reduced step.

4. **Form reduced matrices**
   - `tmpB = B*J`
   - `Ared = J'*AJ` (i.e. `J'*(A*J)`), symmetric up to roundoff
   - `Bred = J'*tmpB` (i.e. `J'*(B*J)`), symmetric up to roundoff

5. **Auto-rank selection inside reduced space**
   - Diagonalize `Ared`: `FA = eigen!(Symmetric(Ared))` (ascending)
   - Threshold: `thr = max(eps_rank*dmax, eps(T))`, with `dmax = d[end]`
   - Keep the last `r` eigenvalues `dkeep` and eigenvectors `Ukeep` with `d > thr`
     (these span the numerical range in the reduced subspace).

6. **Whiten + standard eigenproblem**
   - `Tmat = Ukeep * diag(1/sqrt(dkeep))`
   - Form `B2 = Tmat' * Bred * Tmat`
   - Return `eigvals!(Symmetric(B2))`
"""
function generalized_eig_cholesky(A::Symmetric{T,Matrix{T}},B::Symmetric{T,Matrix{T}};eps_rank::T=T(1e-15),want_vecs::Bool=false) where {T<:Real}
    pad=50
    Fchol,_=_cholpiv_factor(A;eps_rank=eps_rank)
    r0=Fchol.rank
    r0<=0 && error("cholpiv rank r0=0. Decrease eps_rank.")
    n=size(A,1); m=min(n,r0+pad)
    J =Matrix{T}(undef,n,m)
    AJ=Matrix{T}(undef,n,m)
    _cholpiv_fill!(J,Fchol;r0=r0)
    if m>r0
        rng=Random.default_rng()
        @views randn!(rng, J[:,r0+1:m])
    end
    _thin_qr!(J)
    _improve_subspace!(J,AJ,A)
    tmpB=Matrix{T}(undef,n,m)
    Ared=Matrix{T}(undef,m,m)
    Bred=Matrix{T}(undef,m,m)
    mul!(tmpB,B,J)
    _sym_trans_X_Y!(Ared,J,AJ)
    _sym_trans_X_Y!(Bred,J,tmpB)
    FA=eigen!(Symmetric(Ared))
    d=FA.values;U=FA.vectors
    dmax=d[end]
    thr=max(eps_rank*dmax, eps(T))
    r=count(>(thr), d)
    r==0 && error("auto-rank selected r=0. Decrease eps_rank.")
    i0=m-r+1
    Ukeep=Matrix{T}(undef,m,r)
    dkeep=Vector{T}(undef,r)
    @inbounds for j in 1:r
        i=i0+(j-1)
        dkeep[j]=d[i]
        @views Ukeep[:,j].=U[:,i]
    end
    Tmat=copy(Ukeep)
    @inbounds for j in 1:r
        @views Tmat[:,j].*=inv(sqrt(dkeep[j]))
    end
    tmp_mr=Matrix{T}(undef,m,r)
    B2=Matrix{T}(undef,r,r)
    mul!(tmp_mr,Bred,Tmat)
    mul!(B2,transpose(Tmat),tmp_mr)
    if !want_vecs
        return eigvals!(Symmetric(B2))
    end
    S=Matrix{T}(undef,n,r)
    mul!(S,J,Tmat)
    F2=eigen!(Symmetric(B2))
    return F2.values,F2.vectors,S
end