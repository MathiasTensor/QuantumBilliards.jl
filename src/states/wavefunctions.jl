"""
    wavefunction_multi(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno},ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::vec_bdPoints::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096,use_float_32::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. The integrand used is the SLP kernel, so the wavefunction is given by `Ψ(x,y) = (1/4) ∮ Y₀(k|q-q_s|) u(s) ds` where `q` are the boundary points and `u` is the boundary density. The kernel is real, so if `u` is real the wavefunction will be real as well, and if `u` is complex the wavefunction will be complex as well.

# Arguments
- `solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno}`: DLP solver, either naive or Kress. Also supports Vergini Saraceno, since it is solving the same boundary integral equation with the result of it's `boundary_function`.
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` objects, one for each eigenvalue.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Union{Float64,Symbol}=:auto`: (Optional), Point scaling factor. Default is :auto. If not the same as in solver then artifacts can potentially emerge. Reason it is left as kwarg is to speed up wavefunction matrix construction.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread
- `use_float_32::Bool=true`: (Optional), Whether to compute the wavefunction using Float32 Bessel evaluations for speed. Default is true. This can be used when the number of points in the grid is very large and the Bessel evaluations become a bottleneck. For most cases this is most useful.

# Returns
- `Psi2ds::Vector{Matrix{eltype(vec_us[1])}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
"""
function wavefunction_multi(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental=true,MIN_CHUNK=4_096,use_float_32::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    k_max=maximum(ks)
    type=eltype(k_max)
    L=billiard.length
    b= b==:auto ? (typeof(solver.pts_scaling_factor)<:Real ? solver.pts_scaling_factor : solver.pts_scaling_factor[1]) : b
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    end
    dx,dy=xlim[2]-xlim[1],ylim[2]-ylim[1]
    nx,ny=max(round(Int,k_max*dx*b/(2*pi)),512),max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid,y_grid=collect(type,range(xlim..., nx)),collect(type,range(ylim..., ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    sz=length(pts)
    # Determine points inside the billiard only once if inside_only is true
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=fundamental) : fill(true,sz)
    pts_masked_indices=findall(pts_mask)
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    S=eltype(vec_us[1])<:Real ? type : Complex{type}
    Psi_flat=zeros(S,nx*ny) # overwritten each iteration since pts_masked_indices is the same for each k in ks
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k,bdPoints,us=ks[i],vec_bdPoints[i],vec_us[i]
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                # compute this thread's block [lo:hi]
                lo=(t-1)*q+min(t-1,r) + 1
                hi=lo+q-1+(t<=r ? 1 : 0)
                @inbounds for jj in lo:hi
                    idx=pts_masked_indices[jj] # each interior point [idx] -> (x,y)
                    x,y=pts[idx]
                    Psi_flat[idx]=use_float_32 ? ϕ_slp_float32_bessel(x,y,k,bdPoints,us) : ϕ_slp(x,y,k,bdPoints,us)
                end
            end
        end
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
end

"""
    wavefunction_multi_with_husimi(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno},ks::Vector{T},vec_us::Vector{Vector{T}},vec_bdPoints::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto, inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096,use_float_32::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Constructs a sequence of 2D wavefunctions as matrices over the same sized grid for easier computation of matrix elements. The matrices are constructed via the boundary integral. The integrand is the slp kernel, so the wavefunction is given by `Ψ(x,y) = (1/4) ∮ Y₀(k|q-q_s|) u(s) ds` where `q` are the boundary points and `u` is the boundary density. The kernel is real, so if `u` is real the wavefunction will be real as well, and if `u` is complex the wavefunction will be complex as well. Additionally also constructs the husimi functions.

# Arguments
- `solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno}`: DLP solver, either naive or Kress. Also supports Vergini Saraceno, since it is solving the same boundary integral equation with the result of it's `boundary_function`.
- `ks`: Vector of eigenvalues.
- `vec_bdPoints`: Vector of `BoundaryPoints` or `BoundaryPointsCFIE` objects, one for each eigenvalue.
- `billiard`: The billiard geometry.
- `vec_us::Vector{Vector}`: Vector of the boundary functions. Can be either complex or real, this determines whether the wavefunction is real or not.
- `b::Union{Float64,Symbol}=:auto`: (Optional), Point scaling factor. Default is `:auto`. If so it reuses the same point scaling factor as the solver, otherwise it uses the one provided. If not the same as in solver then artifacts can potentially emerge. Reason it is left as kwarg is to speed up wavefunction matrix construction.
- `inside_only::Bool=true`: (Optional), Whether to only compute wavefunctions inside the billiard. Default is true.
- `fundamental::Bool=true`: (Optional), Whether to use fundamental domain for boundary integral. Default is true.
- `xgrid_size::Int=2000`: (Optional), Size of the x grid for the husimi functions. Default is 2000.
- `ygrid_size::Int=1000`: (Optional), Size of the y grid for the husimi functions. Default is 1000.
- `use_fixed_grid::Bool=true`: (Optional), Whether to use a fixed grid for the husimi functions. Default is true.
- `MIN_CHUNK::Int=4096`: keep ≥ this many boundary points per thread
- `use_float_32::Bool=true`: (Optional), Whether to compute the wavefunction using Float32 Bessel evaluations for speed. Default is true.
- `full_p::Bool=false`: (Optional), Whether to compute the full p grid for the husimi functions. Default is false which is the conservative route when one does not know which irrep gives p -> -p symmetry.

# Returns
- `Psi2ds::Vector{Matrix{eltype(vec_us[1])}}`: Vector of 2D wavefunction matrices constructed on the same grid.
- `x_grid::Vector{T}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{T}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix{T}}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector{T}}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector{T}}`: Vector of qs grids for the husimi matrices.
"""
function wavefunction_multi_with_husimi(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,VerginiSaraceno},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental=true,use_fixed_grid=true,xgrid_size=2000,ygrid_size=1000,MIN_CHUNK=4_096,use_float_32::Bool=true,full_p::Bool=false) where {Bi<:AbsBilliard,T<:Real}
    Psi2ds,x_grid,y_grid=wavefunction_multi(solver,ks,vec_us,vec_bdPoints,billiard;b=b,inside_only=inside_only,fundamental=fundamental,MIN_CHUNK=MIN_CHUNK,use_float_32=use_float_32)
    if use_fixed_grid
        Hs_list,ps_list,qs_list=husimi_functions_from_us_and_boundary_points(ks,vec_us,vec_bdPoints,xgrid_size,ygrid_size;full_p=full_p)
    else
        vec_of_s_vals=[boundary_s(bdPoints) for bdPoints in vec_bdPoints]
        Hs_list,ps_list,qs_list=husimi_functions_from_boundary_functions(ks,vec_us,vec_of_s_vals;full_p=full_p)
    end
    return Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list
end

#####################################################################
################## CFIE WAVEFUNCTION CONSTRUCTION ###################
#####################################################################

# Flatten the CFIE_kress boundary points into a single cache for faster wavefunction reconstruction, and then evaluate the CFIE_kress wavefunction at many points from the flattened cache and boundary density `u`.
struct CFIEWavefunctionCache{T<:Real}
    x::Vector{T} # boundary x_j
    y::Vector{T} # boundary y_j
    tx::Vector{T} # tangent x-component
    ty::Vector{T} # tangent y-component
    sj::Vector{T} # |tangent_j|
    w::Vector{T} # quadrature weight w_j
    hmin::T
end

"""
    build_cfie_wavefunction_cheb_plan(k, cache, x_grid, y_grid;
        npanels=256, M=16, grading=:uniform,
        rmin_factor=0.7, rmax_pad=1.1)

Build the Chebyshev interpolation plan used during CFIE wavefunction
postprocessing.

The plan stores piecewise-Chebyshev approximations of

    H₀⁽¹⁾(k r),    H₁⁽¹⁾(k r)

on a radial interval `[rmin,rmax]` large enough to cover all distances between
the plotting grid and the boundary nodes.

# Inputs
- `k`:
  Real wavenumber of the state.
- `cache`:
  Flattened CFIE wavefunction cache containing boundary nodes, tangents,
  quadrature weights, and the minimum boundary spacing `hmin`.
- `x_grid`, `y_grid`:
  Cartesian plotting grid coordinates.

# Keyword arguments
- `npanels`:
  Number of radial Chebyshev panels.
- `M`:
  Chebyshev degree on each panel.
- `grading`:
  Radial panel grading. For wavefunction plotting this is usually `:uniform`.
- `rmin_factor`:
  Safety factor multiplying the smaller of plotting-grid spacing and boundary
  spacing to define the interpolation lower radius.
- `rmax_pad`:
  Padding factor applied to the plotting-box diagonal.

# Returns
- `CFIEWavefunctionChebPlan{T}`:
  A pair of Chebyshev Hankel plans for `H₀` and `H₁`.
"""
function build_cfie_wavefunction_cheb_plan(k::T,cache::CFIEWavefunctionCache{T},x_grid::AbstractVector{T},y_grid::AbstractVector{T};npanels::Int=256,M::Int=16,grading::Symbol=:uniform,rmin_factor::T=T(0.7),rmax_pad::T=T(1.1)) where {T<:Real}
    hx=length(x_grid)>1 ? abs(x_grid[2]-x_grid[1]) : one(T)
    hy=length(y_grid)>1 ? abs(y_grid[2]-y_grid[1]) : one(T)
    hgrid=min(hx,hy) # plotting-grid resolution
    hbdry=cache.hmin # boundary-node arc spacing
    rmin=max(rmin_factor*min(hgrid,hbdry),T(1e-12))
    dx=maximum(x_grid)-minimum(x_grid)
    dy=maximum(y_grid)-minimum(y_grid)
    rmax=rmax_pad*hypot(dx,dy)
    h0=plan_h(0,1,Float64(k),Float64(rmin),Float64(rmax);npanels=npanels,M=M,grading=grading)
    h1=plan_h(1,1,Float64(k),Float64(rmin),Float64(rmax);npanels=npanels,M=M,grading=grading)
    return CFIEWavefunctionChebPlan{T}(h0,h1,T(rmin),T(rmax))
end

"""
    chebyshev_params(k, cache, x_grid, y_grid;
        n_panels_init=15000, M_init=5, grading=:uniform,
        tol=1e-12, sampling_points=20000, max_iter=20,
        grow_panels=1.5, grow_M=2, geo_ratio=1.05,
        rmin_factor=0.5, rmax_pad=1.01, verbose=false)

Tune the Chebyshev parameters for CFIE wavefunction reconstruction.

This function tests piecewise-Chebyshev approximations of

    H₀⁽¹⁾(k r),    H₁⁽¹⁾(k r)

against direct special-function evaluation on a sampled radial interval. It
increases either the number of panels or the polynomial degree until both
maximum errors are below `tol`, or until `max_iter` is reached.

This is intended to be run once at the largest `k` in a batch. The resulting
`n_panels` and `M` can then be reused for all smaller wavenumbers in the same
plotting job.

# Inputs
- `k`:
  Wavenumber used for tuning, usually `maximum(ks)`.
- `cache`:
  Flattened CFIE cache for the state associated with `k`.
- `x_grid`, `y_grid`:
  Cartesian plotting grid coordinates.

# Keyword arguments
- `n_panels_init`:
  Initial number of radial Chebyshev panels.
- `M_init`:
  Initial Chebyshev degree per panel.
- `grading`:
  Radial panel grading, usually `:uniform`.
- `tol`:
  Maximum allowed interpolation error for both `H₀` and `H₁`.
- `sampling_points`:
  Number of radial sample points used for the error check.
- `max_iter`:
  Maximum tuning iterations.
- `grow_panels`:
  Multiplicative panel-growth factor when increasing panel count.
- `grow_M`:
  Additive degree increment used every fifth iteration.
- `geo_ratio`:
  Geometric grading ratio, only relevant if `grading != :uniform`.
- `rmin_factor`:
  Safety factor applied to the boundary spacing when estimating the lower
  radius.
- `rmax_pad`:
  Padding factor applied to the plotting-box diagonal.
- `verbose`:
  If `true`, print worst-error diagnostics at each tuning step.

# Returns
- `n_panels::Int`:
  Tuned number of Chebyshev panels.
- `M::Int`:
  Tuned Chebyshev degree.
- `plan0::ChebHankelPlanH`:
  Final interpolation plan for `H₀⁽¹⁾(k r)`.
- `plan1::ChebHankelPlanH`:
  Final interpolation plan for `H₁⁽¹⁾(k r)`.
- `max_err0`:
  Final maximum interpolation error for `H₀`.
- `max_err1`:
  Final maximum interpolation error for `H₁`.
"""
function chebyshev_params(k::T,cache::CFIEWavefunctionCache{T},x_grid::AbstractVector{T},y_grid::AbstractVector{T};n_panels_init::Int=15_000,M_init::Int=5,grading::Symbol=:uniform,tol::Real=1e-12,sampling_points::Int=20_000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,geo_ratio::Real=1.05,rmin_factor::T=T(0.5),rmax_pad::T=T(1.01),verbose::Bool=false) where {T<:Real}
    hbdry=cache.hmin
    rmin=max(rmin_factor*hbdry,T(1e-12))
    dx=maximum(x_grid)-minimum(x_grid)
    dy=maximum(y_grid)-minimum(y_grid)
    rmax=rmax_pad*hypot(dx,dy)
    rmin_cheb=T(hankel_z_chebyshev_cutoff)/abs(k)
    rmin_interp=max(Float64(rmin),Float64(rmin_cheb))
    @info "Estimated CFIE wavefunction Chebyshev radial bounds" rmin rmax rmin_cheb rmin_interp
    rs=collect(range(Float64(rmin_interp),Float64(rmax);length=sampling_points))
    n_panels=n_panels_init
    M=M_init
    plan0=plan_h(0,1,Float64(k),Float64(rmin_interp),Float64(rmax);npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
    plan1=plan_h(1,1,Float64(k),Float64(rmin_interp),Float64(rmax);npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
    approx0=Vector{ComplexF64}(undef,sampling_points)
    approx1=Vector{ComplexF64}(undef,sampling_points)
    exact0=Vector{ComplexF64}(undef,sampling_points)
    exact1=Vector{ComplexF64}(undef,sampling_points)
    max_err0=Inf
    max_err1=Inf
    for it in 1:max_iter
        plan0=plan_h(0,1,Float64(k),Float64(rmin_interp),Float64(rmax);npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
        plan1=plan_h(1,1,Float64(k),Float64(rmin_interp),Float64(rmax);npanels=n_panels,M=M,grading=grading,geo_ratio=geo_ratio)
        Threads.@threads for i in eachindex(rs)
            r=rs[i]
            p0=_find_panel(plan0,r)
            P0=plan0.panels[p0]
            t0=(2*r-(P0.b+P0.a))/(P0.b-P0.a)
            p1=_find_panel(plan1,r)
            P1=plan1.panels[p1]
            t1=(2*r-(P1.b+P1.a))/(P1.b-P1.a)
            approx0[i]=eval_h(plan0,Int32(p0),t0,r)
            approx1[i]=eval_h(plan1,Int32(p1),t1,r)
            z=ComplexF64(k)*r
            exact0[i]=SpecialFunctions.besselh(0,1,z)
            exact1[i]=SpecialFunctions.besselh(1,1,z)
        end
        max_err0=maximum(abs.(approx0.-exact0))
        max_err1=maximum(abs.(approx1.-exact1))
        if verbose
            i0=argmax(abs.(approx0.-exact0))
            i1=argmax(abs.(approx1.-exact1))
            @info "CFIE wavefunction Chebyshev tuning" it n_panels M max_err0 max_err1
            @info "Worst H0" r=rs[i0] z=ComplexF64(k)*rs[i0] err=abs(approx0[i0]-exact0[i0])
            @info "Worst H1" r=rs[i1] z=ComplexF64(k)*rs[i1] err=abs(approx1[i1]-exact1[i1])
        end
        if max_err0<tol && max_err1<tol
            return n_panels,M,plan0,plan1,max_err0,max_err1
        end
        if it%5==0
            M+=grow_M
        else
            n_panels=ceil(Int,grow_panels*n_panels)
        end
    end
    @warn "CFIE wavefunction Chebyshev tuning did not reach tol=$tol after $max_iter iterations. Returning best effort." max_err0 max_err1 n_panels M
    return n_panels,M,plan0,plan1,max_err0,max_err1
end

"""
    flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real} -> CFIEWavefunctionCache{T}

Contains just enough information to evaluate the CFIE wavefunction at many points without needing 
to reconstruct the `BoundaryPointsCFIE` objects or do any extra computations.

# Inputs:
- `comps`: Vector of `BoundaryPointsCFIE` objects, one for each component of the boundary.

# Outputs:
- `CFIEWavefunctionCache{T}`: A struct containing flattened vectors of boundary coordinates, tangents, quadrature weights, etc., for efficient CFIE wavefunction evaluation.
"""
function flatten_cfie_wavefunction_cache(comps::Vector{BoundaryPointsCFIE{T}}) where {T<:Real}
    N=sum(length(c.xy) for c in comps)       # total number of boundary nodes over all components
    x=Vector{T}(undef,N)                     # flattened boundary x-coordinates x_j
    y=Vector{T}(undef,N)                     # flattened boundary y-coordinates y_j
    tx=Vector{T}(undef,N)                    # flattened tangent x-components γ'_x(t_j)
    ty=Vector{T}(undef,N)                    # flattened tangent y-components γ'_y(t_j)
    sj=Vector{T}(undef,N)                    # speed factors |γ'(t_j)|
    w=Vector{T}(undef,N)                     # quadrature weights in parameter variable t
    p=1                                      # global flattened index
    @inbounds for c in comps # loop over boundary components
        for j in eachindex(c.xy) # loop over nodes of this component
            q=c.xy[j] # boundary point q_j = (x_j,y_j)
            t=c.tangent[j] # tangent vector γ'(t_j)
            txj=t[1]  
            tyj=t[2] 
            x[p]=q[1]                       
            y[p]=q[2]                   
            tx[p]=txj                   
            ty[p]=tyj                 
            sj[p]=sqrt(txj*txj+tyj*tyj) # store speed |γ'(t_j)|
            w[p]=c.ws[j] # store parameter quadrature weight Δt_j
            p+=1                         
        end
    end
    hmin=typemax(T)
    @inbounds for i in 1:N
        hmin=min(hmin,w[i]*sj[i])
    end
    return CFIEWavefunctionCache(x,y,tx,ty,sj,w,hmin) 
end

"""
    ϕ_cfie(xp::T, yp::T, k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false,use_chebyshev::Bool=false,cheb::Union{CFIEWavefunctionChebPlan{T},Nothing}=nothing) where {T<:Real} -> Complex{T}

Evaluate the CFIE reconstructed wavefunction at point `(xp, yp)` from a
flattened boundary cache and boundary density `u`.

This uses the same kernel as the CFIE assembly:

    ψ(x) = -∑_j w_j u_j [ (i k / 2) * inn * H1(k r) / r + i k * (i/2) * H0(k r) * s_j ]

where
    inn = t_y (x-x_j) - t_x (y-y_j)

# Arguments
- `xp, yp` : evaluation point p = SVector(xp, yp)
- `k::T`      : real wavenumber
- `cache::CFIEWavefunctionCache{T}`  : flattened CFIE geometry cache
- `u::AbstractVector{Complex{T}}`      : complex boundary density, same ordering as flattening
- `float32_bessel::Bool`     : evaluate Hankels in Float32 and cast back
- `use_chebyshev::Bool`      : use Chebyshev interpolation evaluation for Hankel functions
- `cheb::Union{CFIEWavefunctionChebPlan{T},Nothing}` : precomputed Chebyshev plan for Hankel evaluation, required if `use_chebyshev=true`
# Returns
- `Complex{T}`: the reconstructed wavefunction value at (xp, yp)
"""
@inline function ϕ_cfie(xp::T,yp::T,k::T,cache::CFIEWavefunctionCache{T},u::AbstractVector{Complex{T}};float32_bessel::Bool=false,use_chebyshev::Bool=false,cheb::Union{CFIEWavefunctionChebPlan{T},Nothing}=nothing) where {T<:Real}
    x=cache.x
    y=cache.y
    tx=cache.tx
    ty=cache.ty
    sj=cache.sj
    w=cache.w
    N=length(x)
    @assert length(u)==N
    ψr=zero(T)
    ψi=zero(T)
    # Constants:
    # dterm = (i k / 2) * inn * H1 / r
    # sterm = (i / 2) * H0 * sj
    # contribution = -(w*u) * (dterm + i k * sterm)
    #
    # Since i*k*sterm = i*k*(i/2) H0 sj = -(k/2) H0 sj,
    # the kernel is
    #   K = (i k / 2) * inn * H1 / r  -  (k / 2) * H0 * sj
    khalf=k*T(0.5)
    tol2=(T(0.5)*cache.hmin)^2 # for near boundary skipping since we have log and 1/r singularities
    @inbounds @fastmath for j in 1:N # fastmath since we remove the singular/near-singular region
        dx=xp-x[j]
        dy=yp-y[j]
        r2=muladd(dx,dx,dy*dy)
        r2<=tol2 && continue # skip near-boundary points
        r=sqrt(r2)
        invr=inv(r)
        inn=muladd(ty[j],dx,-(tx[j]*dy)) # ty*dx - tx*dy
        if use_chebyshev
            h0,h1=_eval_h0h1_cfie_cheb(cheb,r)
        else
            if float32_bessel
                zf=Float32(k*r)
                h0=Complex{T}(Bessels.hankelh1(0,zf))
                h1=Complex{T}(Bessels.hankelh1(1,zf))
            else
                z=k*r
                h0=Bessels.hankelh1(0,z)
                h1=Bessels.hankelh1(1,z)
            end
        end
        # Kernel:
        # K = (i k / 2) * inn * H1 / r - (k / 2) * H0 * sj
        #
        # Let A = (k/2) * inn/r, B = (k/2) * sj
        # Then
        #   K = i*A*h1 - B*h0
        A=khalf*inn*invr
        #B=khalf*sj[j]
        B=khalf*sj[j]
        # i*A*h1 = (-A*imag(h1)) + i*(A*real(h1))
        Kr=muladd(-A,imag(h1),-B*real(h0))
        Ki=muladd(A,real(h1),-B*imag(h0))
        # contribution = -(w*u)*K
        uj=u[j]
        wr=w[j]*real(uj)
        wi=w[j]*imag(uj)
        # -(wr+i wi)(Kr+i Ki)
        ψr-= wr*Kr-wi*Ki
        ψi-= wr*Ki+wi*Kr
    end
    return Complex{T}(ψr,ψi)
end

#####################################################################

"""
    function wavefunction_multi(solver::Union{CFIE_kress,CFIE_alpert,CFIE_kress_corners,CFIE_kress_global_corners},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,use_chebyshev::Bool=true,tol_cheb=1e-12,cheb_verbose=true) where {Bi<:AbsBilliard,T<:Real}

Construct a batch of interior wavefunction intensity matrices for CFIE-based solvers
on a common Cartesian grid.

For each wavenumber `k ∈ ks`, the interior wavefunction `ψ(x,y)` is evaluated via
the CFIE representation
    ψ = -(D + i k S) μ,
where `μ` is the CFIE boundary density and `D,S` are the double- and single-layer
potentials. The resulting matrices store `|ψ(x,y)|` evaluated on a shared grid.

The evaluation is restricted to interior points (if `inside_only=true`) and is
parallelized over spatial grid points.

# Arguments
- `solver::Union{CFIE_kress,CFIE_alpert,CFIE_kress_corners,CFIE_kress_global_corners}`  
  CFIE solver used to generate the boundary densities.
- `ks::Vector{T}`  
  Wavenumbers / eigenvalues.
- `vec_us::Vector{<:AbstractVector{<:Number}}`  
  CFIE boundary densities `μ`, one per state. Each vector is concatenated over
  all boundary components.
- `vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}}`  
  Boundary discretizations for each state. Each entry contains the CFIE boundary
  components (possibly multiple connected components).
- `billiard::Bi`  
  Billiard geometry.
# Keyword arguments
- `b::Union{Float64,Symbol}=:auto`  
  Spatial grid density scaling. If `:auto`, uses the solver’s internal scaling.
- `inside_only::Bool=true`  
  If `true`, evaluate only at points inside the billiard domain.
- `fundamental::Bool=false`  
  If `true`, uses fundamental-domain bounding box; otherwise full domain.
- `MIN_CHUNK::Int=4096`  
  Minimum number of grid points per thread chunk.
- `float32_bessel::Bool=true`  
  Use `Float32` kernel evaluations for faster computation.
- `use_chebyshev::Bool=true`
  Use chebyshev interpolation for hankel evaluations. Basically neccesery for lage number of states for large k.
- `tol_cheb=1e-12`
  Accuracy of the interpolated hankel functions. For plotting this does not need to be very high, maybe 1e-12 - 1e-14.
- `cheb_verbose::Bool=true`
  If panelization steps are to be explititely printed as diagnostic information.

# Returns
- `Psi2ds::Vector{Matrix{T}}`  
  Wavefunction intensity matrices `|ψ(x,y)|`, one per state, all defined on the
  same spatial grid.
- `x_grid::Vector{T}`  
  Grid coordinates in the x-direction.
- `y_grid::Vector{T}`  
  Grid coordinates in the y-direction.
"""
function wavefunction_multi(solver::Union{CFIE_kress,CFIE_alpert,CFIE_kress_corners,CFIE_kress_global_corners},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental::Bool=false,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,use_chebyshev::Bool=true,tol_cheb=1e-12,cheb_verbose=true) where {Bi<:AbsBilliard,T<:Real}
    kmax,idx_max=findmax(ks)
    L=billiard.length
    b= b==:auto ? (typeof(solver.pts_scaling_factor)<:Real ? solver.pts_scaling_factor : solver.pts_scaling_factor[1]) : b
    xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,kmax*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/(2π)),512)
    ny=max(round(Int,kmax*dy*b/(2π)),512)
    x_grid=collect(T,range(xlim[1],xlim[2],length=nx))
    y_grid=collect(T,range(ylim[1],ylim[2],length=ny))
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    npts=length(pts)
    pts_mask=inside_only ? points_in_billiard_polygon(pts,billiard,round(Int,sqrt(npts));fundamental_domain=fundamental) : fill(true,npts)
    pts_masked_indices=findall(pts_mask)
    nmask=length(pts_masked_indices)
    NT=Threads.nthreads()
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    S=eltype(vec_us[1])
    nstates=length(ks)
    Psi2ds=Vector{Matrix{S}}(undef,nstates)
    caches=Vector{CFIEWavefunctionCache{T}}(undef,nstates)
    @inbounds for i in 1:nstates
        caches[i]=flatten_cfie_wavefunction_cache(vec_comps[i])
    end
    cheb_plans=use_chebyshev ? Vector{CFIEWavefunctionChebPlan{T}}(undef,nstates) : fill(nothing,nstates)
    if use_chebyshev
        cheb_npanels,cheb_M,plan1,max_err0,max_err1=chebyshev_params(kmax,caches[idx_max],x_grid,y_grid,verbose=cheb_verbose,tol=tol_cheb)
        @inbounds for i in eachindex(ks)
            cheb_plans[i]=build_cfie_wavefunction_cheb_plan(ks[i],caches[i],x_grid,y_grid;npanels=cheb_npanels,M=cheb_M,grading=cheb_grading)
        end
    end
    Psi_flat=zeros(S,nx*ny)
    progress=Progress(nstates,desc="Constructing CFIE wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        cache=caches[i]
        us=vec_us[i]
        fill!(Psi_flat,zero(S))
        @fastmath begin
            Threads.@threads :static for t in 1:NT_eff
                lo=(t-1)*q+min(t-1,r)+1
                hi=lo+q-1+(t<=r ? 1 : 0)
                for jj in lo:hi
                    idx=pts_masked_indices[jj]
                    p=pts[idx]
                    Psi_flat[idx]=ϕ_cfie(p[1],p[2],k,cache,us;float32_bessel=float32_bessel,use_chebyshev=use_chebyshev,cheb=cheb_plans[i])
                end
            end
        end
        Psi2ds[i]=copy(abs.(reshape(Psi_flat,nx,ny)))
        next!(progress)
    end
    for i in eachindex(Psi2ds)
        nrm=sqrt(sum(abs2,Psi2ds[i][pts_masked_indices]))
        Psi2ds[i]./=nrm
    end
    return Psi2ds,x_grid,y_grid
end

"""
    wavefunction_multi_with_husimi(solver::Union{CFIE_kress,CFIE_alpert,CFIE_kress_corners,CFIE_kress_global_corners},ks::Vector{T},vec_μ::Vector{<:AbstractVector{<:Complex{T}}},vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental::Bool=false,xgrid_size::Int=2000,ygrid_size::Int=1000,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,full_p::Bool=false,normalize_components::Bool=true,multithreaded_boundary_function::Bool=true,use_chebyshev::Bool=true,tol_cheb=1e-12,cheb_verbose=true) where {Bi<:AbsBilliard,T<:Real}

Construct interior wavefunctions and component-wise Husimi functions for a batch
of CFIE eigenstates on a common spatial grid.

1. Wavefunction reconstruction (interior)
   For each `k ∈ ks`, the interior wavefunction is evaluated on a common grid via
   the CFIE representation
       ψ = -(D + i k S) μ

2. Boundary function recovery  
   The physical boundary function
       u = ∂ₙψ |_{∂Ω}
   is reconstructed from `μ` using the CFIE boundary operator and normalized
   via the Rellich identity.

3. Husimi construction (per component)
   The boundary is split into connected components (outer boundary + holes), and
   a Husimi function is computed independently on each component using its own
   arclength parametrization.

# Arguments
- `solver`:
  CFIE solver (`CFIE_kress`, `CFIE_alpert`, or corner variants).
- `ks::Vector{T}`:
  Wavenumbers / eigenvalues.
- `vec_μ::Vector{<:AbstractVector{<:Complex{T}}}`:
  CFIE boundary densities, one per state (concatenated across components).
- `vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}}`:
  Boundary discretizations for each state. Entries sharing the same `compid`
  belong to the same connected boundary component.
- `billiard::Bi`:
  Billiard geometry.

# Keyword arguments
- `b::Union{Float64,Symbol}`:
  Grid density scaling (`:auto` uses solver value).
- `inside_only::Bool`:
  Restrict wavefunction evaluation to interior points.
- `fundamental::Bool`:
  Use fundamental-domain bounding box.
- `xgrid_size::Int`, `ygrid_size::Int`:
  Husimi grid resolution.
- `MIN_CHUNK::Int`:
  Minimum number of spatial points per thread.
- `float32_bessel::Bool`:
  Use Float32 kernel evaluations.
- `full_p::Bool`:
  If `false`, use p→−p symmetry; otherwise compute full momentum grid.
- `normalize_components::Bool`:
  Normalize each component Husimi independently.
- `multithreaded_boundary_function::Bool`:
  Enable threading in CFIE boundary reconstruction.
  - `use_chebyshev::Bool=true`
  Use chebyshev interpolation for hankel evaluations. Basically neccesery for lage number of states for large k.
- `tol_cheb=1e-12`
  Accuracy of the interpolated hankel functions. For plotting this does not need to be very high, maybe 1e-12 - 1e-14.
- `cheb_verbose::Bool=true`
  If panelization steps are to be explititely printed as diagnostic information.

# Returns
- `Psi2ds::Vector{Matrix{eltype(vec_μ[1])}}`:
  Wavefunction intensity matrices on a common grid.
- `x_grid::Vector{T}`, `y_grid::Vector{T}`:
  Spatial grid coordinates.
- `Hs_list::Vector{Vector{Matrix{T}}}`:
  Husimi matrices. `Hs_list[i][a]` corresponds to state `i`, component `a`.
- `ps_list::Vector{Vector{T}}`:
  Momentum grids (identical across states).
- `qs_list::Vector{Vector{Vector{T}}}`:
  Position grids per state and component.
- `u_bdry::Vector{Vector{Complex{T}}}`:
  Boundary functions `u = ∂ₙψ`.
- `pts_bdry::Vector{Vector{BoundaryPointsCFIE{T}}}`:
  Boundary discretizations.
- `L_list::Vector{Vector{T}}`:
  Component-wise boundary lengths.
"""
function wavefunction_multi_with_husimi(solver::Union{CFIE_kress,CFIE_alpert,CFIE_kress_corners,CFIE_kress_global_corners},ks::Vector{T},vec_μ::Vector{<:AbstractVector},vec_comps::AbstractVector{<:AbstractVector{BoundaryPointsCFIE{T}}},billiard::Bi;b::Union{Float64,Symbol}=:auto,inside_only::Bool=true,fundamental::Bool=false,xgrid_size::Int=2000,ygrid_size::Int=1000,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,full_p::Bool=false,normalize_components::Bool=true,multithreaded_boundary_function::Bool=true,use_chebyshev::Bool=true,tol_cheb=1e-12,cheb_verbose=true) where {Bi<:AbsBilliard,T<:Real}
    Psi2ds,x_grid,y_grid=wavefunction_multi(solver,ks,vec_μ,vec_comps,billiard;b=b,inside_only=inside_only,fundamental=fundamental,MIN_CHUNK=MIN_CHUNK,float32_bessel=float32_bessel,use_chebyshev=use_chebyshev,tol_cheb=tol_cheb,cheb_verbose=cheb_verbose)
    pts_bdry,u_bdry=boundary_function(solver,vec_μ,vec_comps,billiard,ks;multithreaded=multithreaded_boundary_function)
    Hs_list,ps,qs_list,L_list=husimi_functions_from_us_and_boundary_points(ks,u_bdry,pts_bdry,xgrid_size,ygrid_size;full_p=full_p,normalize_components=normalize_components)
    ps_list=[ps for _ in eachindex(Hs_list)]
    return Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,u_bdry,pts_bdry,L_list
end

###########################################################################
###########################################################################
###########################################################################

function plot_curve!(ax,crv::AbsRealCurve;plot_normal=true,dens=20.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function plot_curve!(ax,crv::AbsVirtualCurve;plot_normal=false,dens=10.0,color_crv=:grey,linewidth=0.75)
    L=crv.length
    grid=max(round(Int,L*dens),3)
    t=range(0.0,1.0,grid)
    pts=curve(crv,t)
    lines!(ax,pts,color=color_crv,linestyle=:dash,linewidth=linewidth)
    if plot_normal
        ns=normal_vec(crv,t)
        arrows!(ax,getindex.(pts,1),getindex.(pts,2),getindex.(ns,1),getindex.(ns,2),color=:black,lengthscale=0.1)
    end
    ax.aspect=DataAspect()
end

function _plot_boundary_object!(ax,boundary;dens=100.0,plot_normal=true,color_crv=:grey,linewidth=0.75)
    if boundary isa AbstractVector
        for obj in boundary
            _plot_boundary_object!(ax,obj;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
        end
    else
        plot_curve!(ax,boundary;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
    end
    return ax
end

function plot_boundary!(ax,billiard::AbsBilliard;fundamental_domain=true,desymmetrized_full_domain=false,dens=100.0,plot_normal=true,color_crv=:grey,linewidth=0.75)
    if fundamental_domain
        boundary=billiard.fundamental_boundary
    elseif desymmetrized_full_domain
        boundary=billiard.desymmetrized_full_boundary
    else
        boundary=billiard.full_boundary
    end
    _plot_boundary_object!(ax,boundary;dens=dens,plot_normal=plot_normal,color_crv=color_crv,linewidth=linewidth)
    return ax
end

"""
    batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)

Splits a large dataset into batches and calls the provided plotting function on each batch. 

This is useful when plotting a large number of wavefunctions or other data items at once would 
either be too large or time-consuming. By batching, you can generate multiple figures, each 
containing a subset of the data.

# Arguments
- `plot_func::Function`: The plotting function to be called for each batch.
- `args...`: The argument lists. The first argument should be a vector (e.g., `ks`) that 
   determines the number of data items. All other arguments must also be indexable and have 
   a compatible length.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed on to `plot_func`.

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each produced by `plot_func` 
   on a batch of data.
"""
function batch_wrapper(plot_func::Function, args...; N::Integer=100, kwargs...)
    # Extract the data vectors and the number of items
    ks=args[1] # ks is always the first argument
    @assert length(ks)>0 "ks cannot be empty."
    num_batches=ceil(Int,length(ks)/N)
    figures=Vector{Figure}(undef,num_batches)
    for i in 1:num_batches
        start_idx=(i-1)*N+1
        end_idx=min(i*N,length(ks))
        range=start_idx:end_idx
        batched_args=map(arg-> 
        if arg isa AbstractVector  # check if the argument is a vector
            if length(arg)==length(ks)  # slice if it matches `ks`
                arg[range]
            else
                arg  # x_grid, y_grid
            end
        else
            arg # legacy for billiard arg, remove later
        end,args)
        figures[i]=plot_func(batched_args...;kwargs...) # Call the original plotting function for the batch
    end
    return figures
end

"""
    partition_vector(ks::Vector{<:Real}, N::Integer)

Partitions the ks::Vector into N chunks. This is a helper function that helps us map the partitioned figures from the plotting functions (which give us Vector{Figure}) with the corresponding k values they contain. It partitians ks=[k1,k2,...,k_m] as vectors of length N [[k1...k_n],[k_n+1,...,k_2n],.... It is compatible with N=1 (just returns each k as a separate 1-element vector) and length(ks) < N, in which case it return the input vector unchanged.

# Arguments
- `ks::Vector{<:Real}`: The vector of eigenvalues to be partitioned.
- `N::Integer`: The number of chunks to partition the ks vector into.

# Returns
- `partitions::Vector{Vector{<:Real}}`: A vector of vectors, each containing N elements from the input ks vector.
"""
function partition_vector(ks::Vector, N::Integer)
    partitions=[ks[i:min(i+N-1,length(ks))] for i in 1:N:length(ks)]
    return partitions
end

# for a nicer plotting of the wavefunction, we can choose to plot either the real part, the imaginary part, the absolute value, or the absolute value squared. This is useful since various irreps can have complex wavefucntions, and thsoe need a better/different way to represent them visually.
@inline function wavefunction_plot_data(ψ::AbstractMatrix;mode::Symbol=:auto)
    if mode===:auto
        mode=eltype(ψ)<:Real ? :real : :abs
    end
    A=
        mode===:real  ? real.(ψ) :
        mode===:imag  ? imag.(ψ) :
        mode===:abs   ? abs.(ψ)  :
        mode===:abs2  ? abs2.(ψ) :
        error("Unknown wavefunction plot mode: $mode. Use :auto, :real, :imag, :abs, :abs2.")
    amax=maximum(abs,A)
    amax>0 && (A./=amax)
    return A
end

"""
    plot_wavefunctions_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,billiard::Bi;b::Float64=5.0,width_ax::Integer=300, height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the wavefunction_multi or a similar function.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `wave_mode::Symbol=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. Default is `:auto`, which plots the real part for real wavefunctions and the absolute value for complex wavefunctions.
- `plt_boundary::Bool=true`: Whether to plot the boundary of the billiard on top of the wavefunction. Default is true.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,billiard::Bi;b::Float64=5.0,width_ax::Integer=300, height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        wm= wave_mode===:auto ? (eltype(Psi2ds[j]) <: Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        heatmap!(ax,x_grid,y_grid,ψplot;colormap= wm in (:real,:imag) ? :balance : Reverse(:gist_heat),colorrange= wm in (:real,:imag) ? (-1,1) : (0,1))
        plt_boundary && plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector{Vector},y_grid::Vector{Vector},billiard::Bi;b::Float64=5.0, width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary). The x_grid and y_grid is supplied from the `wavefunctions` method since it expects for each wavefunctions it's separate x and y grid.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{Vector}`: Vector of x-coordinates for the grid for each wavefunction.
- `y_grid::Vector{Vector}`: Vector of y-coordinates for the grid for each wavefunction.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `wave_mode::Symbol=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. Default is `:auto`, which plots the real part for real wavefunctions and the absolute value for complex wavefunctions.
- `plt_boundary::Bool=true`: Whether to plot the boundary of the billiard on top of the wavefunction. Default is true.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector{Vector},y_grid::Vector{Vector},billiard::Bi;b::Float64=5.0, width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)),size=(round(Int,1.5*width_ax*max_cols),round(Int,2*height_ax*n_rows)))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions..." for j in eachindex(ks)
        title= isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        wm= wave_mode===:auto ? (eltype(Psi2ds[j]) <: Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        heatmap!(ax,x_grid[j],y_grid[j],ψplot;colormap= wm in (:real,:imag) ? :balance : Reverse(:gist_heat),colorrange= wm in (:real,:imag) ? (-1,1) : (0,1))
        plt_boundary && plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,billiard::Bi;N::Integer=100,kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions specified by `Psi2ds` on the domain defined by `x_grid` and `y_grid`, 
for the billiard geometry `billiard`. The eigenvalues are provided in `ks`. When the number of 
wavefunctions is large, this function automatically splits the data into batches of size `N` 
and generates multiple figures.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices corresponding to `ks`.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch. If `length(ks) > N`, multiple figures are produced.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)
- `wave_mode::Symbol=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. Default is `:auto`, which plots the real part for real wavefunctions and the absolute value for complex wavefunctions.

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch.
"""
function plot_wavefunctions(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,billiard::Bi;N::Integer=100,kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions(ks::Vector,Psi2ds::Vector,x_grid::Vector{Vector},y_grid::Vector{Vector},billiard::Bi; N::Integer=100,kwargs...) where {Bi<:AbsBilliard}

Similar to `plot_wavefunctions` above, but this version allows for a distinct `(x_grid, y_grid)` 
for each wavefunction in `Psi2ds`. This is useful if each wavefunction was computed on a different 
grid. Automatically splits the data into batches of size `N` if `ks` is large.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices, one for each `(x_grid[j], y_grid[j])`.
- `x_grid::Vector{Vector{<:Real}}`: A vector of x-coordinate vectors, one per wavefunction.
- `y_grid::Vector{Vector{<:Real}}`: A vector of y-coordinate vectors, one per wavefunction.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: The number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, check _BATCH function)
- `wave_mode::Symbol=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. Default is `:auto`, which plots the real part for real wavefunctions and the absolute value for complex wavefunctions.

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, one per batch of wavefunctions.
"""
function plot_wavefunctions(ks::Vector,Psi2ds::Vector,x_grid::Vector{Vector},y_grid::Vector{Vector},billiard::Bi; N::Integer=100,kwargs...) where {Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_BATCH,ks,Psi2ds,x_grid,y_grid,billiard;N=N,kwargs...)
end

################################################
################ WITH HUSIMIS ##################
################################################

# helper to concatenate multiple Husimi components into one big Husimi matrix with the qs grids concatenated with offsets to avoid overlap, and also return the seam locations for plotting vertical lines to indicate the component boundaries. This is useful for plotting multiple Husimi components together in a single plot while visually separating them along the q-axis.
function _husimi_concat_with_separation(Hs::Vector{<:AbstractMatrix{T}},qs_list::Vector{<:AbstractVector{T}}) where {T<:Real}
    length(Hs)==length(qs_list) || error("Hs and qs_list must have same length")
    (isempty(Hs) || any(isempty,Hs)) && return zeros(T,0,0),T[],T[]
    ncomp=length(Hs)
    nps=size(Hs[1],2)
    for a in 1:ncomp
        size(Hs[a],1)==length(qs_list[a]) || error("Component $a has inconsistent Husimi/q-grid sizes")
        size(Hs[a],2)==nps || error("All Husimi components must share the same p-grid")
    end
    qcat=T[]
    seams=T[]
    Hblocks=Vector{Matrix{T}}(undef,ncomp)
    qoff=zero(T)
    for a in 1:ncomp
        qa=collect(qs_list[a])
        Ha=Matrix{T}(Hs[a])
        if a>1
            push!(seams,qoff)
        end
        qa_shift=qa.+qoff
        append!(qcat,qa_shift)
        Hblocks[a]=Ha
        qoff=isempty(qa_shift) ? qoff : qa_shift[end]
    end
    Hcat=vcat(Hblocks...)
    return Hcat,qcat,seams
end

# helper to compute the breakpoints along the q-axis for plotting vertical lines to indicate the boundaries between different Husimi components when they are concatenated together. This is useful for visually separating the different components in a combined Husimi plot.
function boundary_hole_breakpoints(pts)
    if pts isa Vector
        lengths = [maximum(boundary_s(p)) for p in pts]
        return cumsum(lengths)[1:end-1]
    else
        return Float64[]
    end
end

# helper to plot vertical lines on the Husimi plot to indicate the separation between different Husimi components when they are concatenated together with offsets. This function takes in the axis to plot on, the seam locations, and the p-grid values to determine the y-limits of the lines.
function _plot_husimi_separation_lines!(ax,seams::AbstractVector{T},ps::AbstractVector{T};color=:cyan,linewidth=3,linestyle=:dash) where {T<:Real}
    isempty(seams) && return ax
    ymin=minimum(ps)
    ymax=maximum(ps)
    for s0 in seams
        lines!(ax,[s0,s0],[ymin,ymax],color=color,linewidth=linewidth,linestyle=linestyle)
    end
    return ax
end

"""
    plot_wavefunctions_with_husimi_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,Hs_list::Vector{<:Vector{<:AbstractMatrix}},ps_list::Vector,qs_list::Vector{<:Vector{<:AbstractVector}},billiard::Bi; b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4,plt_boundary::Bool=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `seam_color=:cyan`: The color of the seam lines separating the Husimi components.
- `seam_linewidth=2`: The linewidth of the seam lines separating the Husimi components.
- `wave_mode=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. If `:auto`, it will plot the real part for real-valued wavefunctions and the absolute value for complex-valued wavefunctions.
- `plt_boundary::Bool=true`: Whether to plot the billiard boundary on the wavefunction axes.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},x_grid::AbstractVector{T},y_grid::AbstractVector{T},Hs_list::Vector{<:Union{AbstractMatrix{T},Vector{<:AbstractMatrix{T}}}},ps_list::Vector{<:AbstractVector{T}},qs_list::Vector{<:Union{AbstractVector{T},Vector{<:AbstractVector{T}}}},billiard::Bi;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4,wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,1.5*height_ax*n_rows),size=(3*width_ax*max_cols,1.5*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title=isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        wm= wave_mode===:auto ? (eltype(Psi2ds[j]) <: Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        heatmap!(ax,x_grid,y_grid,ψplot;colormap= wm in (:real,:imag) ? :balance : Reverse(:gist_heat),colorrange= wm in (:real,:imag) ? (-1,1) : (0,1))
        plt_boundary && plot_boundary!(ax,billiard,fundamental_domain=fundamental,plot_normal=false)
        Hs=Hs_list[j];qs=qs_list[j];Hs=Hs isa AbstractMatrix ? [Hs] : Hs;qs=(qs isa AbstractVector{T} && !(eltype(qs)<:AbstractVector)) ? [qs] : qs
        Hcat,qcat,seams=_husimi_concat_with_separation(Hs,qs)
        !isempty(Hcat) && heatmap!(ax_h,qcat,ps_list[j],Hcat;colormap=Reverse(:gist_heat))
        _plot_husimi_separation_lines!(ax_h,seams,ps_list[j];color=seam_color,linewidth=seam_linewidth)
        xlims!(ax,xlim)
        ylims!(ax,ylim)
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi_BATCH(ks::Vector,Psi2ds::Vector,x_grid::Vector,y_grid::Vector,Hs_list::Vector,ps_list::Vector,qs_list::Vector,billiard::Bi,us_all::Vector,pts_all::Vector;b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4,plt_boundary::Bool=true) where {Bi<:AbsBilliard}

Plots the wavefunctions into a grid (only the fundamental boundary) together with the respective husimi function matrices on the provided grids. The x_grid and y_grid is supplied from the wavefunction_multi or a similar function, and the ps and qs grids mudt also be supplied for plotting the Husimi functions. This version also accepts the us boundary functions and the corresponding arclength evaluation point (us_all -> Vector{Vector{T}} and pts_all -> s_vals_all -> Vector{Vector{T}}) that this function was evaluated on.

# Arguments
- `ks`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Vector of 2D wavefunction matrices.
- `x_grid::Vector{<:Real}`: Vector of x-coordinates for the grid.
- `y_grid::Vector{<:Real}`: Vector of y-coordinates for the grid.
- `Hs_list::Vector{Matrix}`: Vector of 2D husimi function matrices.
- `ps_list::Vector{Vector}`: Vector of ps grids for the husimi matrices.
- `qs_list::Vector{Vector}`: Vector of qs grids for the husimi matrices.
- `billiard<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Vector of us boundary functions.
- `pts_all::Vector{Vector{T}}`: Vector of solver dependant boundary points that the us_all functions were evaluated on.
- `b::Float64=5.0`: The point scaling factor.
- `width_ax::Integer=500`: The size of each axis in the grid layout.
- `height_ax::Integer=500`: The size of each axis in the grid layout.
- `max_cols::Integer=6`: The maximum number of columns in the grid layout.
- `fundamental::Bool=true`: If plotting just the desymmetrized part.
- `custom_label::Vector{String}`: The labels to be plotted for each Axis in the Figure. ! Needs to be the same length as ks, as it should be unique to each k in ks !.
- `seam_color=:cyan`: The color of the seam lines separating the Husimi components.
- `seam_linewidth=2`: The linewidth of the seam lines separating the Husimi components.
- `wave_mode=:auto`: The mode for plotting the wavefunction. Can be `:auto`, `:real`, `:imag`, `:abs`, or `:abs2`. If `:auto`, it will plot the real part for real-valued wavefunctions and the absolute value for complex-valued wavefunctions.
- `plt_boundary::Bool=true`: Whether to plot the billiard boundary on the wavefunction axes.

 # Returns
- `f::Figure`: A Figure object containing the grid of wavefunctions.
"""
function plot_wavefunctions_with_husimi_BATCH(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},x_grid::AbstractVector{T},y_grid::AbstractVector{T},Hs_list::Vector{<:Union{AbstractMatrix{T},Vector{<:AbstractMatrix{T}}}},ps_list::Vector{<:AbstractVector{T}},qs_list::Vector{<:Union{AbstractVector{T},Vector{<:AbstractVector{T}}}},billiard::Bi,us_all::Vector{<:AbstractVector{<:Number}},pts_all::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T},Vector{BoundaryPointsCFIE{T}}}};b::Float64=5.0,width_ax::Integer=300,height_ax::Integer=300,max_cols::Integer=6,fundamental=true,custom_label::Vector{String}=String[],seam_color=:cyan,seam_linewidth=4,wave_mode::Symbol=:auto,plt_boundary::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    L=billiard.length
    if fundamental
        xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    else
        xlim,ylim=boundary_limits(billiard.full_boundary;grd=max(1000,round(Int,maximum(ks)*L*b/(2*pi))))
    end
    n_rows=ceil(Int,length(ks)/max_cols)
    f=Figure(resolution=(3*width_ax*max_cols,2*height_ax*n_rows),size=(3*width_ax*max_cols,2*height_ax*n_rows))
    row=1
    col=1
    @showprogress desc="Plotting wavefunctions and husimi..." for j in eachindex(ks)
        title=isempty(custom_label) ? "$(ks[j])" : custom_label[j]
        local ax_wave=Axis(f[row,col][1,1],title=title,aspect=DataAspect(),width=width_ax,height=height_ax)
        wm= wave_mode===:auto ? (eltype(Psi2ds[j]) <: Real ? :real : :abs) : wave_mode
        ψplot=wavefunction_plot_data(Psi2ds[j];mode=wm)
        heatmap!(ax_wave,x_grid,y_grid,ψplot;colormap= wm in (:real,:imag) ? :balance : Reverse(:gist_heat),colorrange= wm in (:real,:imag) ? (-1,1) : (0,1))
        plt_boundary && plot_boundary!(ax_wave,billiard,fundamental_domain=fundamental,plot_normal=false)
        xlims!(ax_wave,xlim)
        ylims!(ax_wave,ylim)
        local ax_h=Axis(f[row,col][1,2],width=width_ax,height=height_ax)
        Hs=Hs_list[j];qs=qs_list[j];Hs=Hs isa AbstractMatrix ? [Hs] : Hs;qs=(qs isa AbstractVector{T} && !(eltype(qs)<:AbstractVector)) ? [qs] : qs
        Hcat,qcat,seams=_husimi_concat_with_separation(Hs,qs)
        pgrid=ps_list[j]
        !isempty(Hcat) && heatmap!(ax_h,qcat,pgrid,Hcat;colormap=Reverse(:gist_heat))
        _plot_husimi_separation_lines!(ax_h,seams,pgrid;color=seam_color,linewidth=seam_linewidth)
        local ax_boundary=Axis(f[row,col][2,1:2],xlabel="s",ylabel="u(s)",width=2*width_ax,height=height_ax/2)
        svals=boundary_s(pts_all[j])
        breakpoints=boundary_hole_breakpoints(pts_all[j])
        lines!(ax_boundary,svals,real.(us_all[j]),label="Re u(s)",linewidth=2)
        if maximum(abs.(imag.(us_all[j])))>0
            lines!(ax_boundary,svals,imag.(us_all[j]),label="Im u(s)",linewidth=2,linestyle=:dash)
        end
        if !isempty(breakpoints)
            vlines!(ax_boundary,breakpoints,color=seam_color,linewidth=seam_linewidth,linestyle=:dash)
        end
        col+=1
        if col>max_cols
            row+=1
            col=1
        end
    end
    return f
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their corresponding Husimi distributions. Automatically 
splits large datasets into batches of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices associated with each wavefunction.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinate grids for the Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinate grids for the Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects with wavefunction and Husimi plots, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},x_grid::AbstractVector{T},y_grid::AbstractVector{T},Hs_list::Vector{<:Union{AbstractMatrix{T},Vector{<:AbstractMatrix{T}}}},ps_list::Vector{<:AbstractVector{T}},qs_list::Vector{<:Union{AbstractVector{T},Vector{<:AbstractVector{T}}}},billiard::Bi;N::Integer=100,kwargs...) where {T<:Real,Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard;N=N,kwargs...)
end

"""
    plot_wavefunctions_with_husimi(ks::Vector, Psi2ds::Vector, x_grid::Vector, y_grid::Vector, Hs_list::Vector, ps_list::Vector, qs_list::Vector, billiard::Bi, us_all::Vector, pts_all::Vector; N=100, kwargs...) where {Bi<:AbsBilliard}

Plots the wavefunctions along with their Husimi distributions and boundary functions `us_all` 
evaluated at `pts_all -> s_vals_all`. This function also handles a large number of wavefunctions by batching 
the data into sets of size `N`.

# Arguments
- `ks::Vector`: Vector of eigenvalues.
- `Psi2ds::Vector{Matrix}`: Wavefunction matrices.
- `x_grid::Vector{<:Real}`: X-coordinates for the wavefunction grid.
- `y_grid::Vector{<:Real}`: Y-coordinates for the wavefunction grid.
- `Hs_list::Vector{Matrix}`: Husimi function matrices.
- `ps_list::Vector{Vector{<:Real}}`: Momentum-like coordinates for Husimi functions.
- `qs_list::Vector{Vector{<:Real}}`: Position-like coordinates for Husimi functions.
- `billiard::Bi<:AbsBilliard`: The billiard geometry.
- `us_all::Vector{Vector{T}}`: Boundary functions.
- `pts_all::Vector{Vector{T}}`: Points at which the boundary functions are evaluated.
- `N::Integer=100`: Number of items per batch.
- `kwargs...`: Additional keyword arguments passed to the underlying plotting function (N axes per Figure and custom label, also chaotic PS overlays, check _BATCH function)

# Returns
- `figures::Vector{Figure}`: A vector of `Figure` objects, each containing wavefunction, Husimi plots, and boundary functions, one per batch.
"""
function plot_wavefunctions_with_husimi(ks::Vector{T},Psi2ds::Vector{<:AbstractMatrix},x_grid::AbstractVector{T},y_grid::AbstractVector{T},Hs_list::Vector{<:Union{AbstractMatrix{T},Vector{<:AbstractMatrix{T}}}},ps_list::Vector{<:AbstractVector{T}},qs_list::Vector{<:Union{AbstractVector{T},Vector{<:AbstractVector{T}}}},billiard::Bi,us_all::Vector{<:AbstractVector{<:Number}},pts_all::Vector{<:Union{BoundaryPoints{T},BoundaryPointsCFIE{T},Vector{BoundaryPointsCFIE{T}}}};N::Integer=100,kwargs...) where {T<:Real,Bi<:AbsBilliard}
    batch_wrapper(plot_wavefunctions_with_husimi_BATCH,ks,Psi2ds,x_grid,y_grid,Hs_list,ps_list,qs_list,billiard,us_all,pts_all;N=N,kwargs...)
end

###################################################################################
################# WAVEFUNCTION CONSTRUTION FOR BASIS TYPE METHODS #################
###################################################################################

"""
    compute_psi(vec::Vector, k::T, billiard::Bi, basis::Ba, x_grid, y_grid; inside_only=true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

Computs the wavefunction as a `Matrix` on a grid formed by the vectors `x_grid` and `y_grid`. This is a lower level function for wrappers that require the construction of a wavefunction from a vector of linear expansion coefficients and being constructed on a common grid.

# Arguments
- `vec::Vector{T}`: A vector of coefficients representing the linear expansion coefficients of the wavefunction.
- `k::T`: The k-eigenvalue at which the wavefunction is evaluated.
- `billiard<:AbsBilliard`: An instance of the abstract billiard type representing the physical billiard.
- `basis<:AbsBasis`: An instance of the abstract basis type representing the linear expansion basis.
- `x_grid::Vector{T}`: A vector of x-coordinates at which the wavefunction should be evaluated.
- `y_grid::Vector{T}`: A vector of y-coordinates at which the wavefunction should be evaluated.
# Keyword arguments
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the matrix construction.

# Returns
- `Psi::Matrix{T}`: A matrix representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
"""
function compute_psi(vec::Vector,k::T,billiard::Bi,basis::Ba,x_grid::Vector,y_grid::Vector;inside_only=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}
    eps=set_precision(vec[1])
    sz=length(x_grid)*length(y_grid)
    pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
    if inside_only
        pts_mask=points_in_billiard_polygon(pts,billiard,round(Int,sqrt(sz));fundamental_domain=true)
        pts=pts[pts_mask]
    end
    n_pts=length(pts)
    type=eltype(vec)
    memory=sizeof(type)*basis.dim*n_pts #estimate max memory needed for the matrices
    Psi=zeros(type,sz)
    if memory<memory_limit
        B=basis_matrix(basis,k,pts)
        Psi_pts=B*vec
        if inside_only
            Psi[pts_mask].=Psi_pts
        else
            Psi.=Psi_pts
        end
    else
        println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceded $(Base.format_bytes(memory)).")
        if inside_only
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi[pts_mask].+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        else
            for i in eachindex(vec)
                if abs(vec[i])>eps 
                    Psi.+=vec[i].*basis_fun(basis,i,k,pts)
                end
            end
        end
    end
    return Psi
end

"""
    compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}

Constructs the wavefunction as a `Matrix` from an `Eigenstate` struct on a grid of vectors `x_grid` and `y_grid`.

# Arguments
- `state::S`: An `Eigenstate` struct with a `vec` field representing the wavefunction, a `k_basis` field representing the wavefunction basis, a `basis` field representing the basis set, a `billiard` field representing the billiard.
- `x_grid::Vector{T}`: A vector of `x` coordinates on which to evaluate the wavefunction.
- `y_grid::Vector{T}`: A vector of `y` coordinates on which to evaluate the wavefunction.
- `inside_only::Bool` (optional, default `true`): If `true`, only evaluate the wavefunction inside the billiard.
- `memory_limit::Real` (optional, default `10.0e9`): The maximum memory limit in bytes to use for constructing the wavefunction. If the memory required exceeds this multithreading is disabled.

# Returns
- `Psi::Matrix`: A `Matrix` representing the wavefunction evaluated on the grid.
"""
function compute_psi(state::S,x_grid::Vector{T},y_grid::Vector{T};inside_only=true,memory_limit=10.0e9) where {S<:AbsState,T<:Real}
    compute_psi(state.vec,state.k,state.billiard,state.basis,x_grid,y_grid;inside_only,memory_limit)
end

"""
    wavefunction(vec::Vector, k::T, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}     

Computes the wavefunction matrix and the x and y grids for heatmap plotting. It is contructed from the vec=X[i] of `StateData` and not directly from `StateData`.

# Arguments
- `vec::Vector{<:Real}`: The vector of coefficients of the basis expansion of the wavefunction. It's length determines the resizeing of the `basis`.
- `k<:Real`: The wavenumber for that vec = X[i].
- `billiard<:AbsBilliard`: The billiard geometry.
- `basis<:AbsBasis`: The basis used for constructing the wavefunction from `vec`. Must be the same as the one used for constructing `vec`.
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunction(vec::Vector,k::T,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis,T<:Real}     
    dim=length(vec)
    dim=rescale_dimension(basis,dim)
    basis=resize_basis(basis,billiard,dim,k)
    symmetries=basis.symmetries
    type=eltype(vec)
    L=billiard.length
    xlim,ylim=boundary_limits(billiard.fundamental_boundary;grd=max(1000,round(Int,k*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k*dx*b/(2*pi)),512)
    ny=max(round(Int,k*dy*b/(2*pi)),512)
    x_grid::Vector{type}=collect(type,range(xlim...,nx))
    y_grid::Vector{type}=collect(type,range(ylim...,ny))
    Psi::Vector{type}=compute_psi(vec,k,billiard,basis,x_grid,y_grid;inside_only=inside_only,memory_limit=memory_limit) 
    Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
    x_axis=hasproperty(billiard,:x_axis) ? billiard.x_axis : 0.0
    y_axis=hasproperty(billiard,:y_axis) ? billiard.y_axis : 0.0
    return fundamental_domain ? (Psi2d,x_grid,y_grid) : reflect_wavefunction(Psi2d,x_grid,y_grid,symmetries;x_axis=x_axis,y_axis=y_axis)
end

"""
    wavefunction(state::S; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {S<:AbsState}

Constructs the wavefunction from a given state object (like Eigenstate).

# Arguments
- `state::S`: An instance of the abstract state type representing the state from which the wavefunction should be constructed.
- `b::Float64=5.0`: A scaling factor for the billiard size.
- `inside_only::Bool=true`: If true, only points inside the billiard are considered for evaluation.
- `fundamental_domain::Bool=true`: If true, the wavefunction is computed on the fundamental domain of the billiard.
- `memory_limit=10.0e9`: A limit on the memory usage for the computation in bytes. If the memory usage exceeds this limit, multithreading is disabled for the construction.

# Returns
- `Psi2d::Array{T,2}`: A 2D array representing the wavefunction evaluated on the grid formed by the vectors x_grid and y_grid.
- `x_grid::Vector{T}`: A Vector of x values where the matrix was evaluated.
- `y_grid::Vector{T}`: A Vector of y values where the matrix was evaluated.
"""
function wavefunction(state::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {S<:AbsState}    
    wavefunction(state.vec,state.k,state.billiard,state.basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
end

"""
    wavefunctions(X::Vector, ks::Vector, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis, T<:Real}

High level wrapper for moer efficiently computing wavefunction matrices and the grids for plotting.

# Arguments
- `X::Vector`: A vector of coefficients of the basis expansion of the wavefunction for each k in ks.
- `ks::Vector`: A vector of wavenumbers for which to compute the wavefunction.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `vec_Psi::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `vec_xs::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `vec_ys::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(X::Vector,ks::Vector,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    vec_Psi=Vector{Matrix}(undef,length(ks))
    vec_xs=Vector{Vector}(undef,length(ks))
    vec_ys=Vector{Vector}(undef,length(ks))
    p = Progress(length(ks),1)
    Threads.@threads for i in eachindex(ks) 
        vec=X[i]
        k=ks[i]
        Psi2d,x_grid,y_grid=wavefunction(vec,k,billiard,basis;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        vec_Psi[i]=Psi2d
        vec_xs[i]=x_grid
        vec_ys[i]=y_grid
        next!(p)
    end
    return vec_Psi,vec_xs,vec_ys
end

"""
    wavefunctions(state_data::StateData, billiard::Bi, basis::Ba; b=5.0, inside_only=true, fundamental_domain = true, memory_limit = 10.0e9) :: Tuple{Vector, Vector{Matrix}, Vector{Vector}, Vector{Vector}} where {Bi<:AbsBilliard, Ba<:AbsBasis}

High level wrapper for constructing the wavefunctions as a a `Tuple` of `Vector`s : `Tuple (ks::Vector, Psi2ds::Vector{Matrix}, x_grid::Vector{Vector}, y_grid::Vector{Vector})`.

# Arguments
- `state_data::StateData`: Object containing the wavenumbers, tensions and the coefficients of the wavefunction expansion as a vector of vectors for each k in ks.
- `billiard::Bi`: An object representing the billiard.
- `basis::Ba`: An object representing the basis (rpw, cafb...).
- `b`: The point scalling factor. Default is 5.0.
- `inside_only::Bool`: If true, only the points inside the billiard are considered. Default is true.
- `fundamental_domain::Bool`: If true, the wavefunction information is only constructed in the fundamental domain. Default is true.
- `memory_limit`: The maximum amount of memory (in bytes) for constructing the wavefunction with julia broadcasting operations and the use of the `basis_matrix`. Otherwise we use the `basis_fun` directly. Default is 10.0e9.

# Returns
- `ks::Vector{Float64}`: A vector of wavenumbers.
- `Psi2ds::Vector{Matrix}`: A vector of `Matrix` containing the wavefunction for each k in ks.
- `x_grids::Vector{Vector}`: A vector of `Vector` containing the x grid for each k in ks.
- `y_grids::Vector{Vector}`: A vector of `Vector` containing the y grid for each k in ks.
"""
function wavefunctions(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9) where {Bi<:AbsBilliard, Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    Psi2ds=Vector{Matrix{eltype(ks)}}(undef,length(ks))
    x_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    y_grids=Vector{Vector{eltype(ks)}}(undef,length(ks))
    for i in eachindex(ks) 
        vec=X[i] # vector of vectors
        dim=length(vec)
        dim=rescale_dimension(basis,dim)
        new_basis=resize_basis(basis,billiard,dim,ks[i])
        state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
        Psi2d,x_grid,y_grid=wavefunction(state;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit)
        Psi2ds[i]=Psi2d
        x_grids[i]=x_grid
        y_grids[i]=y_grid
    end
    return ks,Psi2ds,x_grids,y_grids
end

"""
    wavefunction(state::BasisState; xlim =(-2.0,2.0), ylim=(-2.0,2.0), b=5.0) 

Construct the wavefunction for a given basis function defined from a `BasisState` object. It is useful for visualizing the varius basis functions in the chosen basis.

# Arguments
- `state::BasisState`: An object representing the basis function.
- `xlim::Tuple{Float64,Float64}`: The range of x values for the wavefunction. Default is `(-2.0, 2.0)`.
- `ylim::Tuple{Float64,Float64}`: The range of y values for the wavefunction. Default is `(-2.0, 2.0)`.
- `b::Float64`: The point scalling factor. Default is 5.0.

# Returns
- `Psi2d::Array{Float64,2}`: The 2D wavefunction matrix for the given basis matrix.
- `x_grid::Vector{<:Real}`: The x grid formed from the `xlim`.
- `y_grid::Vector{<:Real}`: The y grid formed from the `ylim`.
"""
function wavefunction(state::BasisState;xlim =(-2.0,2.0),ylim=(-2.0,2.0),b=5.0) 
    let k=state.k,basis=state.basis      
        type=eltype(state.vec)
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,k*dx*b/(2*pi)),512)
        ny=max(round(Int,k*dy*b/(2*pi)),512)
        x_grid::Vector{type}=collect(type,range(xlim...,nx))
        y_grid::Vector{type}=collect(type,range(ylim...,ny))
        pts_grid=[SVector(x,y) for y in y_grid for x in x_grid]
        Psi::Vector{type}=basis_fun(basis,state.idx,k,pts_grid) 
        Psi2d::Array{type,2}=reshape(Psi,(nx,ny))
        return Psi2d,x_grid,y_grid
    end
end