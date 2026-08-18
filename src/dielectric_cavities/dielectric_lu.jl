#=
For A(k)=[P(k) Q(k);R(k) T(k)],

the disconnected dielectric interiors give

P=diag_a[χ_aS_aa(n_ak)], Q=diag_a[D_aa(n_ak)-I_a],

while the common exterior couples different cavity boundaries through
R=χ_out*S_ext(n_out k), T=D_ext(n_out k)+I (Wiersig's 2002 paper).

For B=[f;g] in AX=B with X=[φ,ψ],
Pφ+Qψ=f, Rφ+Tψ=g,

and elimination with Q gives

Y=Q⁻¹P, S=R-TY, z=Q⁻¹f, φ=S⁻¹(g-Tz), ψ=z-Yφ.

This requires Q⁻¹ to be nonsingular, which should be the case of the real axis
since we are not solving for interior resonances (so imk<0 should be ok, we can always move nodes of axis)

# Under symmetry, the reduced interior remains block diagonal because every
# target equation still belongs to one definite physical dielectric interior.
# A reduced block contains all active representatives whose full-space nodes lie
# on the same physical cavity. Pure inter-cavity symmetry retains a complete
# representative cavity; mixed intra/inter-cavity symmetry may retain only a
# symmetry-reduced subset of that cavity:
# P_red=diag_α(P_α),   Q_red=diag_α(Q_α).

The reduced exterior remains fully coupled. So with the variables above
we just replace the matrix elements in blocks with 
The implementation is destructive (but this is by design so ok) since A is 
not directly needed anymore:

A11 block α -> Y_α=Q_α⁻¹P_α, (P is needed above)
A12 block α -> LU(Q_α), (action of Q^-1 is needed above)
A21 -> S=R-TY -> LU(S), (action of S^-1 is needed above)
A22 -> T unchanged. (T is needed above)
Ref: Matrix Computations (CS 6210), Bindel, Fall 2012; Week 4: Wednesday, Sep 12; Schur complements
=#

abstract type AbstractWiersigLU{T<:Real} end

"""
    WiersigDenseLU

Dense fallback factorization of the active Wiersig matrix. It is used whenever
the active problem contains only one disconnected-interior block.
"""
struct WiersigDenseLU{T<:Real,F}<:AbstractWiersigLU{T}
    factor::F
end

"""
    WiersigBlockLU

In-place block-Schur factorization of a multi-cavity Wiersig matrix

A=[P Q;R T].

The interior matrices satisfy

    P=diag_α(P_α),   Q=diag_α(Q_α),

where α labels the distinct physical dielectric interiors represented in the
active basis. Without symmetry these are the physical cavities themselves.
Under pure inter-cavity symmetry a complete representative cavity forms each
block; under mixed intra/inter-cavity symmetry the corresponding block may be
smaller because the representative cavity is itself symmetry reduced.

The overwritten matrix stores

A11[I_α,I_α]=Y_α=Q_α⁻¹P_α,
A12[I_α,I_α]=LU(Q_α),
A21=LU(S), S=R-TQ⁻¹P,
A22=T.

`ranges` contains the active matrix range I_α of every independent interior
block. The OG matrix must remain alive because the local and Schur LU
factorizations reference its storage.
"""
struct WiersigBlockLU{T<:Real,FQ,FS}<:AbstractWiersigLU{T}
    A::Matrix{Complex{T}}
    qfactors::FQ
    sfactors::FS
    ranges::Vector{UnitRange{Int}}
    N::Int
end

@inline issuccess(F::WiersigDenseLU)=issuccess(F.factor)
@inline issuccess(F::WiersigBlockLU)=issuccess(F.sfactors)&&all(issuccess,F.qfactors)

"""
    _wiersig_lu_ranges(ws)

Return the active matrix ranges of the independent disconnected dielectric
interior blocks.

Without symmetry, each physical cavity Γ_a contributes one block

    I_a=offs[a]:offs[a+1]-1.

With symmetry, `geom.Ifund` contains the full-space representative node of every
active reduced boundary unknown. The physical cavity of representative `b` is

    a_b=global_to_block[Ifund[b]].

Reduced representatives belonging to the same physical cavity form one
disconnected-interior block. This remains valid for pure inter-cavity symmetry
and for mixed intra/inter-cavity symmetry, where only part of a representative
physical cavity may survive in the reduced basis.
"""
function _wiersig_lu_ranges(ws::AbstractWiersigGeometryWorkspace)
    geom=ws.geom
    if !(geom isa WiersigMultiGeometry)
        return [1:boundary_size(ws)] # single physical dielectric interior
    end
    C=length(geom.offs)-1
    if isnothing(geom.symmetry)
        return [geom.offs[a]:(geom.offs[a+1]-1) for a in 1:C] # one full block per physical cavity
    end
    # Reduced unknown b is represented by full physical node Ifund[b].
    # Group consecutive representatives by the physical cavity containing them.
    reps=geom.Ifund
    Nred=length(reps)
    Nred==geom.Nred||throw(DimensionMismatch("Ifund has $Nred representatives but geometry reports Nred=$(geom.Nred)"))
    ranges=UnitRange{Int}[]
    first=1
    while first<=Nred
        cavity=geom.global_to_block[reps[first]]
        last=first
        while last<Nred&&geom.global_to_block[reps[last+1]]==cavity
            last+=1
        end
        push!(ranges,first:last)
        first=last+1
    end
    return ranges
end

"""
    _wiersig_lu_block_count(ws)

Return the number of independent disconnected-interior blocks in the active
Wiersig matrix.

Without symmetry this is the number of physical cavities. Under symmetry it is
the number of physical cavities represented by the reduced target basis, not
the symmetry-group order. In particular mixed intra/inter-cavity symmetry may
reduce the size of an interior block without changing the number of active
interior blocks.
"""
@inline _wiersig_lu_block_count(ws::AbstractWiersigGeometryWorkspace)=length(_wiersig_lu_ranges(ws))

"""
    _wiersig_use_block_lu(ws)

Use block-Schur elimination whenever the active Wiersig matrix contains at least
two independent disconnected-interior blocks.
"""
@inline function _wiersig_use_block_lu(ws::AbstractWiersigGeometryWorkspace)
    return _wiersig_lu_block_count(ws)>1
end

"""
    lu!(A,ws)

Factorize the active Wiersig matrix inplace. If the active problem contains
only one disconnected-interior block, ordinary `lu!(A)` is used. Otherwise,

    A=[P Q;R T],   P=diag_α(P_α),   Q=diag_α(Q_α),

where `α` labels physical cavities without symmetry and independent
representative cavity orbits after symmetry reduction, and the algorithm forms

    Y_α=Q_α⁻¹P_α,   S=R-TQ⁻¹P,

directly inside `A`.

With `I_α` the active boundary range of interior block `α`. (see header for info)
"""
function lu!(A::Matrix{Complex{T}},ws::AbstractWiersigGeometryWorkspace;check::Bool=true) where {T<:Real}
    if !_wiersig_use_block_lu(ws)
        @blas_multi_then_1 MAX_BLAS_THREADS F=lu!(A;check=check)
        return WiersigDenseLU{T,typeof(F)}(F)
    end
    N=boundary_size(ws)
    ranges=_wiersig_lu_ranges(ws)
    C=length(ranges)
    Fout=nothing # returning block LU factorization params
    @views begin
        # Pφ+Qψ=f ; Rφ+Tψ=g.
        # P and Q contain disconnected interior operators and are therefore
        # exactly cavity diagonal. R and T describe the common exterior and
        # remain fully coupled.
        P=A[1:N,1:N]
        Q=A[1:N,N+1:2*N]
        R=A[N+1:2*N,1:N]
        Tb=A[N+1:2*N,N+1:2*N]
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            # First cavity, factor Q₁=L₁U₁ (done separately to get type lu! julia type) directly inside the corresponding A12 diagonal block.
            I=ranges[1]
            F1=lu!(Q[I,I];check=check)
            qfactors=Vector{typeof(F1)}(undef,C)
            qfactors[1]=F1
            # Overwrite P₁ by Y₁=Q₁⁻¹P₁.
            # No N₁×N₁ matrix is allocated: the original P₁ block is
            # no longer needed once its transformed value Y₁ has been formed.
            ldiv!(F1,P[I,I])
            # Remaining cavities
            # For every disconnected interior Ω_a,
            # Q_a <- LU(Q_a),
            # P_a <- Y_a=Q_a⁻¹P_a.
            @inbounds for a in 2:C
                I=ranges[a]
                Fa=lu!(Q[I,I];check=check)
                qfactors[a]=Fa
                ldiv!(Fa,P[I,I])
            end
            # Schur complement
            # P now stores the block diagonal matrix
            # Y=Q⁻¹P=diag(Y₁,...,Y_C).
            # Therefore TY[:,I_a]=T[:,I_a]Y_a.
            # We accumulate each contribution directly into the corresponding columns of R:
            # R[:,I_a] <- R[:,I_a]-T[:,I_a]Y_a.
            # After all cavity blocks R = R_original-TQ⁻¹P = S.
            # This blockwise GEMM is important: constructing a dense global Y
            # would waste N² storage and perform work on its known zero
            # off-diagonal blocks.
            α=-one(Complex{T})
            β=one(Complex{T})
            @inbounds for a in 1:C
                I=ranges[a]
                mul!(R[:,I],Tb[:,I],P[I,I],α,β)
            end
            # Global exterior Schur factorization,  R now stores S. 
            # This N×N LU is the only remaining global cubic
            # factorization. Its factors are retained in A21.
            sfactors=lu!(R;check=check)
            Fout=WiersigBlockLU{T,typeof(qfactors),typeof(sfactors)}(A,qfactors,sfactors,ranges,N)
        end
    end
    return Fout
end

"""
    ldiv!(X,F,B)

Solve the factorized Wiersig system `AX=B` without modifying `B`
where F is the factorization object holding the factors.

For the block-Schur factorization,

B=[f;g],
Y=diag(Q_a⁻¹P_a),
S=R-TY,

we need to solve for

z=Q⁻¹f,
φ=S⁻¹(g-Tz),
ψ=z-Yφ,

with final output `X=[φ;ψ]`.
"""
function ldiv!(X::Matrix{Complex{T}},F::WiersigDenseLU{T},B::Matrix{Complex{T}}) where {T<:Real}
    copyto!(X,B)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        ldiv!(F.factor,X)
    end
    return X
end

function ldiv!(X::Matrix{Complex{T}},F::WiersigBlockLU{T},B::Matrix{Complex{T}}) where {T<:Real}
    N=F.N
    A=F.A
    @views begin
        # The backing matrix after lu! has the storage interpretation
        # A11 diagonal blocks -> Y_a=Q_a⁻¹P_a,
        # A12 diagonal blocks -> LU(Q_a),
        # A21 -> LU(S),
        # A22 -> original T.
        # Only Y_a and T are needed explicitly during the solve; the local Q
        # and global S operations are accessed through their LU objects.
        Y=A[1:N,1:N]
        Tb=A[N+1:2*N,N+1:2*N]
        f=B[1:N,:]
        g=B[N+1:2*N,:]
        φ=X[1:N,:]
        ψ=X[N+1:2*N,:]
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            # Local interior solve
            # Copy f into the lower-half workspace and solve independently:
            # z_a=Q_a⁻¹f_a.
            # Since Q is block diagonal, the work is Σ_a O(N_a² nrhs) instead of O(N² nrhs).
            copyto!(ψ,f)
            @inbounds for a in eachindex(F.ranges)
                I=F.ranges[a]
                ldiv!(F.qfactors[a],ψ[I,:]) # z_a
            end
            # Schur right-hand side
            # Form h=g-Tz directly in the upper half of X:
            copyto!(φ,g) # φ <- g,
            mul!(φ,Tb,ψ,-one(Complex{T}),one(Complex{T})) # φ <- φ-Tψ.
            # Global Schur solve φ=S⁻¹h.
            # The upper half of X now already contains its final value.
            ldiv!(F.sfactors,φ)
            # Interior back-substitution
            @inbounds for a in eachindex(F.ranges)
                I=F.ranges[a]
                mul!(ψ[I,:],Y[I,I],φ[I,:],-one(Complex{T}),one(Complex{T})) # Since Y=diag(Y_a), ψ_a <- z_a-Y_aφ_a.
            end
        end
    end
    return X
end

"""
    ldiv!(x::Vector{Complex{T}},F::WiersigDenseLU{T},b::Vector{Complex{T}}) where {T<:Real}

Vector-right-hand-side version of `ldiv!`. See the matrix version for more info.
The mathematics is identical to the matrix-RHS method (where x -> X is matrix and not a vector).
This path is useful for diagnostics and individual solves; Beyn should normally use the matrix method so
that many probe vectors are processed together by BLAS-3.
"""
function ldiv!(x::Vector{Complex{T}},F::WiersigBlockLU{T},b::Vector{Complex{T}}) where {T<:Real}
    N=F.N
    A=F.A
    @views begin
        Y=A[1:N,1:N]
        Tb=A[N+1:2*N,N+1:2*N]
        f=b[1:N]
        g=b[N+1:2*N]
        φ=x[1:N]
        ψ=x[N+1:2*N]
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            # z=Q⁻¹f.
            copyto!(ψ,f)
            @inbounds for a in eachindex(F.ranges)
                I=F.ranges[a]
                ldiv!(F.qfactors[a],ψ[I])
            end
            # h=g-Tz.
            copyto!(φ,g)
            mul!(φ,Tb,ψ,-one(Complex{T}),one(Complex{T}))
            # φ=S⁻¹h.
            ldiv!(F.sfactors,φ)
            # ψ_a=z_a-Y_aφ_a.
            @inbounds for a in eachindex(F.ranges)
                I=F.ranges[a]
                mul!(ψ[I],Y[I,I],φ[I],-one(Complex{T}),one(Complex{T}))
            end
        end
    end
    return x
end
# just the dense fallback
function ldiv!(x::Vector{Complex{T}},F::WiersigDenseLU{T},b::Vector{Complex{T}}) where {T<:Real}
    copyto!(x,b)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        ldiv!(F.factor,x)
    end
    return x
end