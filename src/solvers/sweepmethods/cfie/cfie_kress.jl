# helpers for construct_matrices! to determine the bool kwarg for cfie_geom_cache to decide whether to include the log(w') correction
@inline _is_kress_graded(::CFIE_kress,pts::Vector{<:BoundaryPointsCFIE})=false
@inline _is_kress_graded(::CFIE_kress_corners,pts::Vector{<:BoundaryPointsCFIE})=any(_is_nontrivial_grading,pts)
@inline _is_kress_graded(::CFIE_kress_global_corners,pts::Vector{<:BoundaryPointsCFIE})=any(_is_nontrivial_grading,pts)

"""
    CFIEKressWorkspace{T,M}

Reusable cache for CFIE-Kress matrix assembly on a fixed set of boundary
discretization nodes.

This workspace collects all data that depend only on the boundary geometry and
node placement, but not on the current wavenumber `k`. Since repeated sweeps,
determinant scans, Newton refinement, and EBIM-style eigensolvers may assemble
the operator many times for different `k`, it is wasteful to rebuild this
geometry-dependent information each time.

The workspace therefore stores:
- the global block-diagonal Kress correction matrix,
- per-component geometry caches,
- per-component panel-array views,
- component offsets into the global matrix,
- total matrix size.

Mathematical role
-----------------
Suppose the boundary has `nc` connected components. For each component `Γ_a`,
the Kress discretization produces:
- a local logarithmic correction block `R_a`,
- a local geometry cache `G_a`,
- local flat coordinate/tangent arrays `parr_a`.

The full global operator is assembled on the direct sum of all components, so
the global unknown vector is ordered by component. The workspace defines the
mapping between local component indices and the global matrix through `offs`.

# Fields
- `offs::Vector{Int}`:
  Component offsets into the global matrix. If component `a` occupies rows and
  columns `offs[a]:(offs[a+1]-1)`, then `offs[end]-1 == Ntot`.
- `Rmat::M`:
  Global block-diagonal Kress correction matrix. Each diagonal block is the
  Kress logarithmic matrix for one component; off-diagonal blocks are zero
  because the logarithmic singularity is present only in self-interactions.
- `Gs::Vector{CFIEGeomCache{T}}`:
  One precomputed geometry cache per boundary component. Each cache stores
  distances, inverse distances, logarithmic terms, speeds, curvatures, and
  oriented inner products needed in the CFIE formulas.
- `parr::Vector{CFIEPanelArrays{T}}`:
  One flat panel-array cache per component, used for efficient indexed access
  in the off-diagonal inter-component loops.
- `Ntot::Int`:
  Total size of the assembled global matrix.
- `symmetry::Union{Nothing,AbsSymmetry}`:
  If the boundary has a known symmetry group, this field stores it. Otherwise, it is `nothing`.
- `Nred::Int`:
  Number of unique indices in the fundamental domain if symmetry is present; otherwise, `Ntot`.
- `Ifund::Vector{Int}`:
  List of global indices that belong to the fundamental domain (i.e., the unique representatives of each symmetry orbit).
- `full_to_fund::Vector{Int}`:
  Maps each global index to its corresponding fundamental domain index.
- `full_to_scale::Vector{Complex{T}}`:
  Maps each global index to the complex scaling factor associated with its symmetry orbit.
- `fund_to_full::Vector{Vector{Int}}`:
  For each fundamental domain index, lists the global indices in its symmetry orbit.
- `fund_to_scale::Vector{Vector{Complex{T}}}`:
  For each fundamental domain index, lists the complex scaling factors for the indices in its symmetry orbit.
- `global_to_block::Vector{Int}`:
  Maps each global index to the corresponding block/component index.
- `global_to_local::Vector{Int}`:
  Maps each global index to the local index within its block/component.
- `reduced_orbits::Vector{ReducedOrbitInfo{T}}`:
  Precomputed information about the reduced symmetry orbits, used for quick matrix assembly and application in the presence of symmetry.
"""
struct CFIEKressWorkspace{T<:Real,M<:AbstractMatrix{T},S}
    offs::Vector{Int}
    Rmat::M
    Gs::Vector{CFIEGeomCache{T}}
    parr::Vector{CFIEPanelArrays{T}}
    Ntot::Int
    symmetry::S
    Nred::Int
    Ifund::Vector{Int}
    full_to_fund::Vector{Int}
    full_to_scale::Vector{Complex{T}}
    fund_to_full::Vector{Vector{Int}}
    fund_to_scale::Vector{Vector{Complex{T}}}
    global_to_block::Vector{Int}
    global_to_local::Vector{Int}
    reduced_orbits::Vector{ReducedOrbitInfo{T}}
end

"""
    build_cfie_kress_workspace(solver, pts)

Build a reusable `CFIEKressWorkspace` for a fixed vector of CFIE boundary
components.

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`:
  Determines which Kress correction family is used. Whether a component is
  actually treated as graded is inferred from its `ws_der`; components with
  `ws_der ≈ 1` use the ordinary smooth-periodic logarithmic cache.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components of the billiard.

# Returns
- `CFIEKressWorkspace{T}`
"""
function build_cfie_kress_workspace(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Rmat=build_Rmat_kress(solver,pts)
    Gs=[cfie_geom_cache(p,_is_nontrivial_grading(p)) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    Ntot=offs[end]-1
    g2c,g2l=global_to_component_local(pts)
    if isnothing(solver.symmetry)
        Ifund=collect(1:Ntot)
        full_to_fund=collect(1:Ntot)
        full_to_scale=ones(Complex{T},Ntot)
        fund_to_full=[[i] for i in 1:Ntot]
        fund_to_scale=[[one(Complex{T})] for _ in 1:Ntot]
        reduced_orbits=build_reduced_orbit_infos(Ifund,full_to_fund,fund_to_full,fund_to_scale,g2c,g2l)
        return CFIEKressWorkspace(offs,Rmat,Gs,parr,Ntot,nothing,Ntot,Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,g2c,g2l,reduced_orbits)
    end
    sym=solver.symmetry
    maps=build_symmetry_maps(pts,sym)
    Ifund=Int[]
    full_to_fund=zeros(Int,Ntot)
    full_to_scale=zeros(Complex{T},Ntot)
    fund_to_full=Vector{Vector{Int}}()
    fund_to_scale=Vector{Vector{Complex{T}}}()
    visited=falses(Ntot)
    @inbounds for g in 1:Ntot
        visited[g] && continue
        orb=apply_symmetry_orbit_from_maps(maps,sym,g)
        inds=orb.indices
        scales=Complex{T}.(orb.scales)
        push!(Ifund,g)
        b=length(Ifund)
        push!(fund_to_full,inds)
        push!(fund_to_scale,scales)
        for q in eachindex(inds)
            gg=inds[q]
            visited[gg]=true
            full_to_fund[gg]=b
            full_to_scale[gg]=scales[q]
        end
    end
    reduced_orbits=build_reduced_orbit_infos(Ifund,full_to_fund,fund_to_full,fund_to_scale,g2c,g2l)
    return CFIEKressWorkspace(offs,Rmat,Gs,parr,Ntot,sym,length(Ifund),Ifund,full_to_fund,full_to_scale,fund_to_full,fund_to_scale,g2c,g2l,reduced_orbits)
end

"""
    build_Rmat_kress(solver::CFIE_kress, pts)

The logarithmic singular part is universal on an equispaced periodic grid and
depends only on the periodic node difference through the kernel

    log(4 sin²((t-s)/2)).

Its Nyström discretization is encoded by the dense Kress matrix `R`. For a
boundary with several disconnected smooth closed components, this correction is
needed only within each component. Therefore the global matrix has block form

    R =
        [ R₁   0   ...   0 ]
        [ 0    R₂  ...   0 ]
        [ ...           ...]
        [ 0    0   ...  R_nc ].

# Arguments
- `solver::CFIE_kress`
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  One `BoundaryPointsCFIE` per connected smooth closed component.

# Returns
- `Rmat::Matrix{T}`:
  Global block-diagonal Kress correction matrix of size `Ntot × Ntot`.
"""
function build_Rmat_kress(solver::CFIE_kress,pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    Rmat=zeros(T,Ntot,Ntot)
    for a in 1:length(pts)
        ra=offs[a]:(offs[a+1]-1)
        kress_R!(@view Rmat[ra,ra])
    end
    return Rmat
end

"""
    build_Rmat_kress(solver::Union{CFIE_kress_corners,CFIE_kress_global_corners}, pts)

Build the global block-diagonal Kress logarithmic correction matrix for
corner-capable CFIE-Kress formulations. Components whose node sets are genuinely
graded should use the corner-compatible Kress block; ungraded smooth-periodic
components may use the ordinary periodic block.

In the graded-corner versions of the Kress method, the computational nodes are
not uniform in the physical parameter. Instead, a graded variable is introduced
so that corner singularities are regularized. The logarithmic singular split
must therefore be represented on the original graded periodic indexing rather
than with the plain smooth-periodic Kress matrix.

For each connected boundary component, the singular logarithmic correction is
therefore encoded by the corner-aware matrix constructed by `kress_R_corner!`.
As in the smooth case, the global correction matrix is block diagonal:

    R =
        [ R₁   0   ...   0 ]
        [ 0    R₂  ...   0 ]
        [ ...           ...]
        [ 0    0   ...  R_nc ].

# Arguments
- `solver::Union{CFIE_kress_corners,CFIE_kress_global_corners}`
- `pts::Vector{BoundaryPointsCFIE{T}}`

# Returns
- `Rmat::Matrix{T}`:
  Global block-diagonal Kress correction matrix. Each component block is chosen
  from `kress_R_corner!` if that component is genuinely graded, otherwise from
  the ordinary smooth-periodic `kress_R!`.
"""
function build_Rmat_kress(solver::Union{CFIE_kress_corners,CFIE_kress_global_corners},pts::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    offs=component_offsets(pts)
    Ntot=offs[end]-1
    Rmat=zeros(T,Ntot,Ntot)
    for a in eachindex(pts)
        ra=offs[a]:(offs[a+1]-1)
        if _is_nontrivial_grading(pts[a])
            kress_R_corner!(@view Rmat[ra,ra])
        else
            kress_R!(@view Rmat[ra,ra])
        end
    end
    return Rmat
end


"""
    construct_matrices!(solver, A, pts, Rmat, Gs, parr, offs, k; multithreaded=true)

Low-level in-place assembly of the CFIE-Kress system matrix for a full
multi-component boundary.
It assembles the global dense matrix

    A(k) = I - ( D(k) + i k S(k) ),

where:
- `D(k)` is the double-layer operator discretization,
- `S(k)` is the single-layer operator discretization,
- both are Kress-corrected on same-component interactions.

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`:
  Any Kress-based CFIE solver.
- `A::AbstractMatrix{Complex{T}}`:
  Preallocated destination matrix of size `Ntot × Ntot`.
- `pts::Vector{BoundaryPointsCFIE{T}}`:
  Boundary discretization for all connected components.
- `Rmat::AbstractMatrix{T}`:
  Global block-diagonal Kress correction matrix.
- `Gs::Vector{CFIEGeomCache{T}}`:
  Per-component geometry caches.
- `parr::Vector{CFIEPanelArrays{T}}`:
  Per-component flat geometry arrays.
- `offs::Vector{Int}`:
  Component offsets into the global matrix.
- `k::T`:
  Real wavenumber.
- `multithreaded::Bool=true`:
  Enables threaded assembly on sufficiently large blocks.

# Returns
- `A`, modified in place.
"""
function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},Gs::Vector{CFIEGeomCache{T}},parr::Vector{CFIEPanelArrays{T}},offs::Vector{Int},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Pa=parr[a]
        Na=length(Pa.X)
        ra=offs[a]:(offs[a+1]-1)
        @inbounds for i in 1:Na
            gi=ra[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            wi=pa.ws[i]
            dval=Complex{T}(wi*κi,zero(T))
            m1=αM1*si
            m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
            sval=Complex{T}(Rmat[gi,gi]*m1,zero(T))+wi*m2
            A[gi,gi]=one(Complex{T})-(dval+ik*sval)
        end
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                wi=pa.ws[i]
                r=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                h0,h1=hankel_pair01(k*r)
                j0=real(h0)
                j1=real(h1)
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                dval_ij=Rmat[gi,gj]*l1_ij+wj*l2_ij
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=Rmat[gi,gj]*m1_ij+wj*m2_ij
                A[gi,gj]=-(dval_ij+ik*sval_ij)
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                dval_ji=Rmat[gj,gi]*l1_ji+wi*l2_ji
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=Rmat[gj,gi]*m1_ji+wi*m2_ji
                A[gj,gi]=-(dval_ji+ik*sval_ji)
            end
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        pa=pts[a]
        pb=pts[b]
        Pa=parr[a]
        Pb=parr[b]
        Na=length(Pa.X)
        Nb=length(Pb.X)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=Pa.X;Ya=Pa.Y
        Xb=Pb.X;Yb=Pb.Y
        dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
        @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            @inbounds for j in 1:Nb
                gj=rb[j]
                dx=xi-Xb[j]
                dy=yi-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=dYb[j]*dx-dXb[j]*dy
                h0,h1=hankel_pair01(k*r)
                sj=sb[j]
                wj=pb.ws[j]
                dval=wj*(αL2*inn*h1*invr)
                sval=wj*(αM2*h0*sj)
                A[gi,gj]=-(dval+ik*sval)
            end
        end
    end
    return A
end

function construct_matrices_reduced!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    Nred=ws.Nred
    @assert size(A,1)==Nred && size(A,2)==Nred
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    @use_threads multithreading=multithreaded for b in 1:Nred
        orb=ws.reduced_orbits[b]
        imgs=orb.full
        scales=orb.scales
        @inbounds for a in 1:Nred
            gi=ws.Ifund[a]
            ib=ws.global_to_block[gi]
            i=ws.global_to_local[gi]
            acc=zero(Complex{T})
            for q in eachindex(imgs)
                gj=imgs[q]
                scale=scales[q]
                jb=ws.global_to_block[gj]
                j=ws.global_to_local[gj]
                if ib==jb
                    p=pts[ib]
                    G=ws.Gs[ib]
                    si=G.speed[i]
                    sj=G.speed[j]
                    wi=p.ws[i]
                    wj=p.ws[j]
                    if i==j
                        dval=Complex{T}(wi*G.kappa[i],zero(T))
                        m1=αM1*si
                        m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
                        sval=Complex{T}(ws.Rmat[gi,gj]*m1,zero(T))+wi*m2
                        acc+=scale*(one(Complex{T})-(dval+ik*sval))
                    else
                        r=G.R[i,j]
                        invr=G.invR[i,j]
                        lt=G.logterm[i,j]
                        inn=G.inner[i,j]
                        h0,h1=hankel_pair01(k*r)
                        j0=real(h0)
                        j1=real(h1)
                        l1=αL1*inn*j1*invr
                        l2=αL2*inn*h1*invr-l1*lt
                        dval=ws.Rmat[gi,gj]*l1+wj*l2
                        m1=αM1*j0*sj
                        m2=αM2*h0*sj-m1*lt
                        sval=ws.Rmat[gi,gj]*m1+wj*m2
                        acc+=scale*-(dval+ik*sval)
                    end
                else
                    Pi=ws.parr[ib]
                    Pj=ws.parr[jb]
                    pb=pts[jb]
                    dx=Pi.X[i]-Pj.X[j]
                    dy=Pi.Y[i]-Pj.Y[j]
                    r2=muladd(dx,dx,dy*dy)
                    if r2>(eps(T))^2
                        r=sqrt(r2)
                        invr=inv(r)
                        inn=Pj.dY[j]*dx-Pj.dX[j]*dy
                        h0,h1=hankel_pair01(k*r)
                        sj=Pj.s[j]
                        wj=pb.ws[j]
                        dval=wj*(αL2*inn*h1*invr)
                        sval=wj*(αM2*h0*sj)
                        acc+=scale*-(dval+ik*sval)
                    end
                end
            end
            A[a,b]=acc
        end
    end
    return A
end

"""
    construct_matrices!(solver, A, A1, A2, pts, Rmat, Gs, parr, offs, k; multithreaded=true)

Low-level in-place assembly of the CFIE-Kress system matrix together with its
first and second derivatives with respect to the wavenumber `k`.

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`
- `A::AbstractMatrix{Complex{T}}`:
  Destination matrix for the operator itself.
- `A1::AbstractMatrix{Complex{T}}`:
  Destination matrix for the first derivative with respect to `k`.
- `A2::AbstractMatrix{Complex{T}}`:
  Destination matrix for the second derivative with respect to `k`.
- `pts::Vector{BoundaryPointsCFIE{T}}`
- `Rmat::AbstractMatrix{T}`
- `Gs::Vector{CFIEGeomCache{T}}`
- `parr::Vector{CFIEPanelArrays{T}}`
- `offs::Vector{Int}`
- `k::T`
- `multithreaded::Bool=true`

# Returns
- `(A, A1, A2)`, modified in place.

# Notes
This is the derivative-aware low-level assembly kernel. Most users will call one
of the higher-level `construct_matrices!` wrappers instead.
"""
function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},Gs::Vector{CFIEGeomCache{T}},parr::Vector{CFIEPanelArrays{T}},offs::Vector{Int},k::T;multithreaded::Bool=true) where {T<:Real}
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    fill!(A1,zero(Complex{T}))
    fill!(A2,zero(Complex{T}))
    nc=length(pts)
    for a in 1:nc
        pa=pts[a]
        Ga=Gs[a]
        Pa=parr[a]
        Na=length(Pa.X)
        ra=offs[a]:(offs[a+1]-1)
        @inbounds for i in 1:Na
            gi=ra[i]
            si=Ga.speed[i]
            κi=Ga.kappa[i]
            wi=pa.ws[i]
            dval=Complex{T}(wi*κi,zero(T))
            m1=αM1*si
            m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
            sval=Complex{T}(Rmat[gi,gi]*m1,zero(T))+wi*m2
            m2_1=-(si/(π*k))
            m2_2=(si/(π*k^2))
            sval1=wi*m2_1
            sval2=wi*m2_2
            A[gi,gi]=one(Complex{T})-(dval+ik*sval)
            A1[gi,gi]=-(Complex{T}(0,1)*sval+ik*sval1)
            A2[gi,gi]=-(Complex{T}(0,2)*sval1+ik*sval2)
        end
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj=ra[j]
            sj=Ga.speed[j]
            wj=pa.ws[j]
            @inbounds for i in 1:(j-1)
                gi=ra[i]
                si=Ga.speed[i]
                wi=pa.ws[i]
                r=Ga.R[i,j]
                invr=Ga.invR[i,j]
                lt=Ga.logterm[i,j]
                inn_ij=Ga.inner[i,j]
                inn_ji=Ga.inner[j,i]
                kr=k*r
                h0,h1=hankel_pair01(kr)
                j0=real(h0)
                j1=real(h1)
                l1_ij=αL1*inn_ij*j1*invr
                l2_ij=αL2*inn_ij*h1*invr-l1_ij*lt
                dval_ij=Rmat[gi,gj]*l1_ij+wj*l2_ij
                l1_ij_1=-(inn_ij*k*j0)*inv_two_pi
                l1_ij_2=(inn_ij*(k*r*j1-j0))*inv_two_pi
                l2_ij_1=(inn_ij*k*(lt*j0+im*π*h0))*inv_two_pi
                l2_ij_2=(inn_ij*(lt*(j0-k*r*j1)+im*π*(h0-k*r*h1)))*inv_two_pi
                dval_ij_1=Rmat[gi,gj]*l1_ij_1+wj*l2_ij_1
                dval_ij_2=Rmat[gi,gj]*l1_ij_2+wj*l2_ij_2
                m1_ij=αM1*j0*sj
                m2_ij=αM2*h0*sj-m1_ij*lt
                sval_ij=Rmat[gi,gj]*m1_ij+wj*m2_ij
                m1_ij_1=(r*sj*j1)*inv_two_pi
                m1_ij_2=(r*sj*(k*r*j0-j1))*inv_two_pi/k
                m2_ij_1=-(r*sj*(lt*j1+im*π*h1))*inv_two_pi
                m2_ij_2=(r*sj*(lt*(j1-k*r*j0)-im*π*k*r*h0+im*π*h1))*inv_two_pi/k
                sval_ij_1=Rmat[gi,gj]*m1_ij_1+wj*m2_ij_1
                sval_ij_2=Rmat[gi,gj]*m1_ij_2+wj*m2_ij_2
                A[gi,gj]=-(dval_ij+ik*sval_ij)
                A1[gi,gj]=-(dval_ij_1+Complex{T}(0,1)*sval_ij+ik*sval_ij_1)
                A2[gi,gj]=-(dval_ij_2+Complex{T}(0,2)*sval_ij_1+ik*sval_ij_2)
                l1_ji=αL1*inn_ji*j1*invr
                l2_ji=αL2*inn_ji*h1*invr-l1_ji*lt
                dval_ji=Rmat[gj,gi]*l1_ji+wi*l2_ji
                l1_ji_1=-(inn_ji*k*j0)*inv_two_pi
                l1_ji_2=(inn_ji*(k*r*j1-j0))*inv_two_pi
                l2_ji_1=(inn_ji*k*(lt*j0+im*π*h0))*inv_two_pi
                l2_ji_2=(inn_ji*(lt*(j0-k*r*j1)+im*π*(h0-k*r*h1)))*inv_two_pi
                dval_ji_1=Rmat[gj,gi]*l1_ji_1+wi*l2_ji_1
                dval_ji_2=Rmat[gj,gi]*l1_ji_2+wi*l2_ji_2
                m1_ji=αM1*j0*si
                m2_ji=αM2*h0*si-m1_ji*lt
                sval_ji=Rmat[gj,gi]*m1_ji+wi*m2_ji
                m1_ji_1=(r*si*j1)*inv_two_pi
                m1_ji_2=(r*si*(k*r*j0-j1))*inv_two_pi/k
                m2_ji_1=-(r*si*(lt*j1+im*π*h1))*inv_two_pi
                m2_ji_2=(r*si*(lt*(j1-k*r*j0)-im*π*k*r*h0+im*π*h1))*inv_two_pi/k
                sval_ji_1=Rmat[gj,gi]*m1_ji_1+wi*m2_ji_1
                sval_ji_2=Rmat[gj,gi]*m1_ji_2+wi*m2_ji_2
                A[gj,gi]=-(dval_ji+ik*sval_ji)
                A1[gj,gi]=-(dval_ji_1+Complex{T}(0,1)*sval_ji+ik*sval_ji_1)
                A2[gj,gi]=-(dval_ji_2+Complex{T}(0,2)*sval_ji_1+ik*sval_ji_2)
            end
        end
    end
    for a in 1:nc, b in 1:nc
        a==b && continue
        pa=pts[a]
        pb=pts[b]
        Pa=parr[a]
        Pb=parr[b]
        Na=length(Pa.X)
        Nb=length(Pb.X)
        ra=offs[a]:(offs[a+1]-1)
        rb=offs[b]:(offs[b+1]-1)
        Xa=Pa.X;Ya=Pa.Y
        Xb=Pb.X;Yb=Pb.Y
        dXb=Pb.dX;dYb=Pb.dY;sb=Pb.s
        @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
            gi=ra[i]
            xi=Xa[i]
            yi=Ya[i]
            @inbounds for j in 1:Nb
                gj=rb[j]
                dx=xi-Xb[j]
                dy=yi-Yb[j]
                r2=muladd(dx,dx,dy*dy)
                r2<=(eps(T))^2 && continue
                r=sqrt(r2)
                invr=inv(r)
                inn=dYb[j]*dx-dXb[j]*dy
                h0,h1=hankel_pair01(k*r)
                sj=sb[j]
                wj=pb.ws[j]
                dval=wj*(Complex{T}(0,k/2)*inn*h1*invr)
                dval1=wj*(-(Complex{T}(0,1)/2)*inn*k*h0)
                dval2=wj*(-(Complex{T}(0,1)/2)*inn*(h0-k*r*h1))
                sval=wj*(Complex{T}(0,1/2)*h0*sj)
                sval1=wj*(-(Complex{T}(0,1)/2)*r*h1*sj)
                sval2=wj*((Complex{T}(0,1)/2)*r*(h1-k*r*h0)*sj/k)
                A[gi,gj]=-(dval+ik*sval)
                A1[gi,gj]=-(dval1+Complex{T}(0,1)*sval+ik*sval1)
                A2[gi,gj]=-(dval2+Complex{T}(0,2)*sval1+ik*sval2)
            end
        end
    end
    return A,A1,A2
end

function construct_matrices_reduced_deriv!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    Nred=ws.Nred
    @assert size(A,1)==Nred && size(A,2)==Nred
    αL1=-k*inv_two_pi
    αL2=Complex{T}(0,k/2)
    αM1=-inv_two_pi
    αM2=Complex{T}(0,one(T)/2)
    ik=Complex{T}(0,k)
    fill!(A,zero(Complex{T}))
    fill!(A1,zero(Complex{T}))
    fill!(A2,zero(Complex{T}))
    @use_threads multithreading=multithreaded for b in 1:Nred
        orb=ws.reduced_orbits[b]
        imgs=orb.full
        scales=orb.scales
        @inbounds for a in 1:Nred
            gi=ws.Ifund[a]
            ib=ws.global_to_block[gi]
            i=ws.global_to_local[gi]
            acc0=zero(Complex{T})
            acc1=zero(Complex{T})
            acc2=zero(Complex{T})
            for q in eachindex(imgs)
                gj=imgs[q]
                scale=scales[q]
                jb=ws.global_to_block[gj]
                j=ws.global_to_local[gj]
                if ib==jb
                    p=pts[ib]
                    G=ws.Gs[ib]
                    si=G.speed[i]
                    sj=G.speed[j]
                    wi=p.ws[i]
                    wj=p.ws[j]
                    if i==j
                        dval=Complex{T}(wi*G.kappa[i],zero(T))
                        m1=αM1*si
                        m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*si^2))*si
                        sval=Complex{T}(ws.Rmat[gi,gj]*m1,zero(T))+wi*m2
                        sval1=wi*(-si/(π*k))
                        sval2=wi*(si/(π*k^2))
                        acc0+=scale*(one(Complex{T})-(dval+ik*sval))
                        acc1+=scale*-(Complex{T}(0,1)*sval+ik*sval1)
                        acc2+=scale*-(Complex{T}(0,2)*sval1+ik*sval2)
                    else
                        r=G.R[i,j]
                        invr=G.invR[i,j]
                        lt=G.logterm[i,j]
                        inn=G.inner[i,j]
                        kr=k*r
                        h0,h1=hankel_pair01(kr)
                        j0=real(h0)
                        j1=real(h1)
                        l1=αL1*inn*j1*invr
                        l2=αL2*inn*h1*invr-l1*lt
                        dval=ws.Rmat[gi,gj]*l1+wj*l2
                        l1_1=-(inn*k*j0)*inv_two_pi
                        l1_2=(inn*(k*r*j1-j0))*inv_two_pi
                        l2_1=(inn*k*(lt*j0+im*π*h0))*inv_two_pi
                        l2_2=(inn*(lt*(j0-k*r*j1)+im*π*(h0-k*r*h1)))*inv_two_pi
                        dval1=ws.Rmat[gi,gj]*l1_1+wj*l2_1
                        dval2=ws.Rmat[gi,gj]*l1_2+wj*l2_2
                        m1=αM1*j0*sj
                        m2=αM2*h0*sj-m1*lt
                        sval=ws.Rmat[gi,gj]*m1+wj*m2
                        m1_1=(r*sj*j1)*inv_two_pi
                        m1_2=(r*sj*(k*r*j0-j1))*inv_two_pi/k
                        m2_1=-(r*sj*(lt*j1+im*π*h1))*inv_two_pi
                        m2_2=(r*sj*(lt*(j1-k*r*j0)-im*π*k*r*h0+im*π*h1))*inv_two_pi/k
                        sval1=ws.Rmat[gi,gj]*m1_1+wj*m2_1
                        sval2=ws.Rmat[gi,gj]*m1_2+wj*m2_2
                        acc0+=scale*-(dval+ik*sval)
                        acc1+=scale*-(dval1+Complex{T}(0,1)*sval+ik*sval1)
                        acc2+=scale*-(dval2+Complex{T}(0,2)*sval1+ik*sval2)
                    end
                else
                    Pi=ws.parr[ib]
                    Pj=ws.parr[jb]
                    pb=pts[jb]
                    dx=Pi.X[i]-Pj.X[j]
                    dy=Pi.Y[i]-Pj.Y[j]
                    r2=muladd(dx,dx,dy*dy)
                    if r2>(eps(T))^2
                        r=sqrt(r2)
                        invr=inv(r)
                        inn=Pj.dY[j]*dx-Pj.dX[j]*dy
                        h0,h1=hankel_pair01(k*r)
                        sj=Pj.s[j]
                        wj=pb.ws[j]
                        dval=wj*(Complex{T}(0,k/2)*inn*h1*invr)
                        dval1=wj*(-(Complex{T}(0,1)/2)*inn*k*h0)
                        dval2=wj*(-(Complex{T}(0,1)/2)*inn*(h0-k*r*h1))
                        sval=wj*(Complex{T}(0,1/2)*h0*sj)
                        sval1=wj*(-(Complex{T}(0,1)/2)*r*h1*sj)
                        sval2=wj*((Complex{T}(0,1)/2)*r*(h1-k*r*h0)*sj/k)
                        acc0+=scale*-(dval+ik*sval)
                        acc1+=scale*-(dval1+Complex{T}(0,1)*sval+ik*sval1)
                        acc2+=scale*-(dval2+Complex{T}(0,2)*sval1+ik*sval2)
                    end
                end
            end
            A[a,b]=acc0
            A1[a,b]=acc1
            A2[a,b]=acc2
        end
    end
    return A,A1,A2
end

"""
    construct_matrices!(solver, A, pts, ws, k; multithreaded=true)
    construct_matrices!(solver, A, pts, Rmat, k; multithreaded=true)
    construct_matrices!(solver, A, A1, A2, pts, ws, k; multithreaded=true)
    construct_matrices!(solver, A, A1, A2, pts, Rmat, k; multithreaded=true)
    construct_matrices!(solver, basis, A, dA, ddA, pts, k; multithreaded=true)
    construct_matrices!(solver, basis, A, dA, ddA, pts, ws, k; multithreaded=true)

High-level CFIE-Kress assembly interface.

# Arguments
Common arguments:
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`
- `A::AbstractMatrix{Complex{T}}`
- `A1::AbstractMatrix{Complex{T}}`, `A2::AbstractMatrix{Complex{T}}`
- `pts::Vector{BoundaryPointsCFIE{T}}`
- `ws::CFIEKressWorkspace{T}`
- `Rmat::AbstractMatrix{T}`
- `basis::AbstractHankelBasis`
- `k::T`
- `multithreaded::Bool=true`

# Returns
- Matrix-only forms return `A`
- Derivative forms return `(A, A1, A2)`
"""
function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(ws.symmetry)
        @blas_1 construct_matrices!(solver,A,pts,ws.Rmat,ws.Gs,ws.parr,ws.offs,k;multithreaded=multithreaded)
    else
        @blas_1 construct_matrices_reduced!(solver,A,pts,ws,k;multithreaded=multithreaded)
    end
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p, _is_nontrivial_grading(p)) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    return construct_matrices!(solver,A,pts,Rmat,Gs,parr,offs,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    if isnothing(ws.symmetry)
        construct_matrices!(solver,A,A1,A2,pts,ws.Rmat,ws.Gs,ws.parr,ws.offs,k;multithreaded=multithreaded)
    else
        construct_matrices_reduced_deriv!(solver,A,A1,A2,pts,ws,k;multithreaded=multithreaded)
    end
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},A::AbstractMatrix{Complex{T}},A1::AbstractMatrix{Complex{T}},A2::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},Rmat::AbstractMatrix{T},k::T;multithreaded::Bool=true) where {T<:Real}
    offs=component_offsets(pts)
    Gs=[cfie_geom_cache(p, _is_nontrivial_grading(p)) for p in pts]
    parr=[_panel_arrays_cache(p) for p in pts]
    return construct_matrices!(solver,A,A1,A2,pts,Rmat,Gs,parr,offs,k;multithreaded=multithreaded)
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},k::T;multithreaded::Bool=true) where {T<:Real}
    ws=build_cfie_kress_workspace(solver,pts)
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

function construct_matrices!(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::AbstractHankelBasis,A::AbstractMatrix{Complex{T}},dA::AbstractMatrix{Complex{T}},ddA::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    construct_matrices!(solver,A,dA,ddA,pts,ws,k;multithreaded=multithreaded)
    return A,dA,ddA
end

"""
    solve(solver, basis, pts, k; multithreaded=true, use_krylov=true, which=:det)
    solve(solver, basis, pts, ws, k; multithreaded=true, use_krylov=true, which=:det_argmin)
    solve(solver, basis, A, pts, ws, k; multithreaded=true, use_krylov=true, which=:det_argmin)

High-level scalar solver interface for the CFIE-Kress family.

Purpose
-------
These methods assemble the global CFIE-Kress matrix and then reduce it to a
single scalar diagnostic, depending on the choice of `which`.

The matrix being analyzed is always

    A(k) = I - ( D(k) + i k S(k) ).

The overloads differ only in how much cached information is reused:

1. `solve(..., pts, k; ...)`
   - builds `Rmat`,
   - allocates the matrix,
   - assembles and reduces.

2. `solve(..., pts, ws, k; ...)`
   - reuses the full workspace,
   - allocates the matrix,
   - assembles and reduces.

3. `solve(..., A, pts, ws, k; ...)`
   - fastest repeated-use form,
   - reuses both matrix storage and full workspace.

Meaning of `which`
------------------
This keyword is forwarded to the generic backend macro `@svd_or_det_solve`.
Typical meanings are:
- `:svd`:
  smallest singular value of `A(k)`,
- `:det`:
  determinant `det(A(k))`,
- `:det_argmin`:
  backend-specific scalar useful for determinant-based minimization.

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`
- `basis::AbsBasis`
- `pts::Vector{BoundaryPointsCFIE{T}}`
- `ws::CFIEKressWorkspace{T}`
- `A::AbstractMatrix{Complex{T}}`
- `k`
- `multithreaded::Bool=true`
- `use_krylov::Bool=true`
- `which::Symbol`

# Returns
A scalar whose meaning depends on `which`.
"""
function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det) where {T<:Real,Ba<:AbsBasis}
    ws=build_cfie_kress_workspace(solver,pts)
    N=isnothing(ws.symmetry) ? ws.Ntot : ws.Nred
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    N=isnothing(ws.symmetry) ? ws.Ntot : ws.Nred
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

function solve(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    @svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
end

#TODO solve_vect should dispath for multi ks version on the chebyshev intepolated matrix construction
"""
    solve_vect(solver, basis, A, pts, ws, k; multithreaded=true, tol=1e-12, maxiter=2000, krylovdim=40)
    solve_vect(solver, basis, pts, ws, k; multithreaded=true, tol=1e-12, maxiter=2000, krylovdim=40)
    solve_vect(solver, billiard, basis, pts, k; multithreaded=true, tol=1e-12, maxiter=2000, krylovdim=40)
    solve_vect(solver, billiard, basis, ks::Vector; multithreaded=true, tol=1e-12, maxiter=2000, krylovdim=40)

Compute the smallest singular value of the CFIE-Kress matrix together with the
corresponding right singular vector.

For a given wavenumber `k`, these methods assemble the global matrix

    A(k) = I - ( D(k) + i k S(k) )

and then compute a full SVD

    A = U Σ V*.

They identify the smallest singular value `μ = σ_min(A)` and return:
- `μ`,
- the corresponding right singular vector.

In the code, this right singular vector is extracted from `Vt` as

    u_mu = conj.(Vt[idx, :])

Overloads
---------
1. `solve_vect(..., A, pts, ws, k)`
   - reuse matrix buffer and full workspace

2. `solve_vect(..., pts, ws, k)`
   - reuse full workspace, allocate matrix

3. `solve_vect(..., pts, k)`
   - fully self-contained form

4. `solve_vect(..., ks::Vector)`
   - batched convenience form over many wavenumbers;
     returns both the singular vectors and the boundary discretizations used at
     each `k`

# Arguments
- `solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners}`
- `basis::AbsBasis` : placeholder `AbstractHankelBasis()`
- `A::AbstractMatrix{Complex{T}}` : preallocated matrix for assembly;
- `pts::Vector{BoundaryPointsCFIE{T}}` : boundary discretization;
  the geometry caches used for assembly will be built from this if not provided
- `ws::CFIEKressWorkspace{T}` : prebuilt workspace containing geometry caches and `Rmat`;
  if this is not provided, it will be built from `pts` (see `build_cfie_kress_workspace`)
- `k` : wavenumber for the single-`k` forms
- `ks::Vector{T}` : vector of wavenumbers for the batched form
- `multithreaded::Bool=true` : whether to use multithreading for assembly;
- `tol=1e-12` : tolerance for convergence of the Krylov solver;
  ignored if `use_krylov=false`
- `maxiter=2000`: maximum number of iterations for the Krylov solver;
  ignored if `use_krylov=false`
- `krylovdim=40`: dimension of the Krylov subspace used for approximating the smallest singular value;
  ignored if `use_krylov=false`

# Returns
Single-`k` forms:
- `mu`:
  smallest singular value
- `u_mu`:
  corresponding right singular vector

Vector-of-`k` form:
- `us_all`:
  singular vectors for each `k`
- `pts_all`:
  discretizations used at each `k`
"""
function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,A::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis}
    N=isnothing(ws.symmetry) ? ws.Ntot : ws.Nred
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},billiard::Bi,basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},k;multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis,Bi<:AbsBilliard}
    ws=build_cfie_kress_workspace(solver,pts)
    N=isnothing(ws.symmetry) ? ws.Ntot : ws.Nred
    A=Matrix{Complex{T}}(undef,N,N)
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    σ,u,_=smallest_nullvec_krylov!(A;nev=1,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
    return σ,u
end

function solve_vect(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},billiard::Bi,basis::Ba,ks::Vector{T};multithreaded::Bool=true,tol=1e-12,maxiter::Int=2000,krylovdim::Int=40) where {T<:Real,Ba<:AbsBasis,Bi<:AbsBilliard}
    us_all=Vector{Vector{eltype(complex(ks[1]))}}(undef,length(ks))
    pts_all=Vector{Vector{BoundaryPointsCFIE{eltype(ks[1])}}}(undef,length(ks))
    for i in eachindex(ks)
        pts=evaluate_points(solver,billiard,ks[i])
        _,u=solve_vect(solver,billiard,basis,pts,ks[i];multithreaded=multithreaded,tol=tol,maxiter=maxiter,krylovdim=krylovdim)
        us_all[i]=u
        pts_all[i]=pts
    end
    return us_all,pts_all
end

# INTERNAL - for benchmarking and diagnostics only; not a public API
function solve_INFO(solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::Ba,pts::Vector{BoundaryPointsCFIE{T}},ws::CFIEKressWorkspace{T},k;multithreaded::Bool=true,use_krylov::Bool=true,which::Symbol=:det_argmin) where {T<:Real,Ba<:AbsBasis}
    N=isnothing(ws.symmetry) ? ws.Ntot : ws.Nred
    A=Matrix{Complex{T}}(undef,N,N)
    t0=time()
    @info "Building boundary operator A from cached Kress workspace..."
    @blas_1 construct_matrices!(solver,A,pts,ws,k;multithreaded=multithreaded)
    t1=time()
    cA=cond(A)
    @info "Condition number of A: $(round(cA; sigdigits=4))"
    t2=time()
    s=@svd_or_det_solve A use_krylov which MAX_BLAS_THREADS
    t3=time()
    build_A=t1-t0
    svd_time=t3-t2
    total=build_A+svd_time
    println("────────── SOLVE_INFO SUMMARY ──────────")
    println("A-matrix build: ",100*build_A/total," %")
    println("SVD: ",100*svd_time/total," %")
    println("(total: ",total," s)")
    println("────────────────────────────────────────")
    return s
end

function plot_boundary_with_weight_INFO(billiard::Bi,solver::Union{CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners};k=20.0,markersize=2,show_arrows=false) where {Bi<:AbsBilliard}
    pts_all=evaluate_points(solver,billiard,k)
    ncomp=length(pts_all)
    f=Figure(resolution=(1100,220*ncomp),fontsize=10)
    axg=Axis(f[1:ncomp,1],aspect=DataAspect(),xticklabelsize=8,yticklabelsize=8)
    sc=nothing
    for pts in pts_all
        xs=getindex.(pts.xy,1)
        ys=getindex.(pts.xy,2)
        ws=pts.ws
        sc=scatter!(axg,xs,ys;markersize=markersize,color=ws,colormap=:viridis,strokewidth=0)
        if show_arrows
            tx=getindex.(pts.tangent,1)
            ty=getindex.(pts.tangent,2)
            arrows!(axg,xs,ys,tx,ty;color=(:black,0.25),lengthscale=0.03)
        end
    end
    Colorbar(f[1,2],sc,width=10) 
    hidespines!(axg,:t,:r)
    for (j,pts) in enumerate(pts_all)
        s=cumsum(pts.ds)
        ws=pts.ws
        ws_der=pts.ws_der
        ws_n=ws./maximum(abs.(ws))
        wsd_l=log10.(abs.(ws_der).+eps())
        ax=Axis(f[j,3],title="c$j",titlesize=10,xticklabelsize=8,yticklabelsize=8)
        lines!(ax,s,ws_n,linewidth=1.8)
        lines!(ax,s,wsd_l,linestyle=:dash,linewidth=1.8)
        hidespines!(ax,:t,:r)
    end
    colgap!(f.layout,2)
    rowgap!(f.layout,2)
    return f
end