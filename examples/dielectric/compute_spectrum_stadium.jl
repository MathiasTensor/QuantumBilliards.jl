using QuantumBilliards
using LinearAlgebra
using Random
using Printf
using Statistics
using CairoMakie

try_MKL!()

# Production compute_spectrum test for a dielectric stadium. The final count is
# compared with the leading dielectric Weyl estimate in one XY symmetry sector.

n_in=1.5                # refractive index inside the cavity
n_out=1.0               # refractive index outside the cavity
ppw=10.0                # boundary points per wavelength
polarization=:TM        # TM vs TE
kmin=135.0              # requested Re(k) interval
kmax=140.0
interval_width=5.0      # tesselate entire region with full geometry reuse (use only 1)
im_min=-0.90            # requested Im(k) strip (far enough from real axis to avoid spurious solutions)
im_max=-0.01
halfwidth=0.4           # template contour half-width in Re(k)
halfheight=0.5          # template contour half-height in Im(k)
use_rectangle_contour=true # best option as some sides can be reused due to tesselation grid
nq=70                   # smooth-contour trapezoidal order; unused when use_rectangle_contour=true
nx_GL=18                # Gauss-Legendre order on each horizontal rectangle edge
ny_GL=18                # Gauss-Legendre order on each vertical rectangle edge (total is 2*nx_GL+2*ny_GL=72)
#TODO This requires some testing (play around!) to see if it is enough for the convergence of the spectrum to normalized_res_tol. It is a trade-off between speed and accuracy. The higher the better, but slower.
svd_tol=collect(1e-6*0.1.^(0:4)) # absolute SVD thresholds generating the candidate reduced-rank ladder
#TODO IMPORTANT: normalized_res_tol is basically a proxy for accuracy (see results from convergence tests in this folder)
normalized_res_tol=1e-11 # normalized nonlinear residual threshold for adaptively checked weak-support candidates
validation_padding=5    # stop after this many consecutive checked weak-support candidates pass
multithreaded=true      # threaded matrix assembly
dlp_kernel=:source      # source-normal DLP convention
chebyshev=true          # keep this test usable on machines with modest RAM
verbose=true
adaptive_validation_method=:linear # less robust than :linear (it is still experimental for a O(log(M)) checking of residuals instead of O(M))
print_spectrum=false    # if the spectrum will be printed to the std::out

################################################################################
################################ GEOMETRY #######################################
################################################################################

# Active test: Bunimovich stadium with both x and y reflections. The (-1,-1) sector contains states odd under both reflections and has asymptotically 1/4 of the full spectrum.
billiard,_=make_stadium_and_basis(0.5)
symmetry=XYReflection(-1,-1)
quadrature_kind=:global_corners

# Smooth comparison geometries:
#billiard,_=make_ellipse_and_basis(1.0,0.5);symmetry=XYReflection(-1,-1);quadrature_kind=:smooth
#billiard,_=make_prosen_and_basis(0.2);symmetry=XYReflection(-1,-1);quadrature_kind=:smooth
solver=WiersigKress(n_in,n_out,billiard,ppw;quadrature_kind=quadrature_kind,polarization=polarization,symmetry=symmetry)

################################################################################
############################ CONTOUR TESSELLATION ##############################
################################################################################

if use_rectangle_contour
    template=WiersigRectangleContour(ComplexF64(0.0,(im_min+im_max)/2),halfwidth,halfheight)
else
    template=wiersig_fourier_rectangle_contour(ComplexF64(0.0,(im_min+im_max)/2),halfwidth,halfheight)
end

@info "COMPUTE_SPECTRUM TEST" geometry=nameof(typeof(billiard)) symmetry=symmetry region=(kmin,kmax,im_min,im_max) nq=(!use_rectangle_contour ? nq : (nx_GL,ny_GL))

################################################################################
############################ COMPUTE SPECTRUM ##################################
################################################################################

kwargs=(nq=(!use_rectangle_contour ? nq : (nx_GL,ny_GL)),svd_tol=svd_tol,relative_svd_tol=false,validate_roots=false,adaptive_validation=true,validation_padding=validation_padding,normalized_res_tol=normalized_res_tol,dlp_kernel=dlp_kernel,multithreaded=multithreaded,chebyshev=chebyshev,verbose=verbose,probe_factor=5.0,adaptive_validation_method=adaptive_validation_method)
tbeyn=@elapsed result=if use_rectangle_contour
    compute_spectrum(solver,kmin,kmax,im_min,im_max,template;interval_width=interval_width,kwargs...)
else
    compute_spectrum(solver,kmin,kmax,im_min,im_max,template;interval_width=interval_width,kwargs...)
end

@info "SPECTRUM SUMMARY" time=round(tbeyn,digits=3) roots=length(result.values) intervals=length(result.interval_results) contours=length(result.contours) Nmin=minimum(result.contour_dimensions) Nmax=maximum(result.contour_dimensions)

################################################################################
############################### FINAL SPECTRUM #################################
################################################################################

# Sort the production spectrum only for presentation. No near-degenerate-state
# merging is performed here.
perm=sortperm(eachindex(result.values);by=i->(real(result.values[i]),imag(result.values[i])))
if print_spectrum
    @info "FINAL SPECTRUM"
    println(rpad("i",6),rpad("Re(k)",17),rpad("Im(k)",17),rpad("residual",14),"contour")
    for (p,i) in enumerate(perm)
        k=result.values[i];nr=result.normalized_residuals[i]
        println(rpad(p,6),rpad(@sprintf("%.12f",real(k)),17),rpad(@sprintf("%.12f",imag(k)),17),rpad(isfinite(nr) ? @sprintf("%.3e",nr) : "—",14),result.source_contours[i])
    end
end

# Reconstruct and plot the 50 resonances with largest Re(k).
nplot=min(50,length(perm));sel=perm[end-nplot+1:end]
ks=result.values[sel];Psi=Vector{Matrix{ComplexF64}}(undef,nplot)
fig=let xg=nothing,yg=nothing,pts0=nothing
    for ic in unique(result.source_contours[sel])
        loc=findall(p->result.source_contours[sel[p]]==ic,eachindex(sel))
        inds=sel[loc];pts=result.contour_pts[ic];ws=result.contour_workspaces[ic]
        # construct the wavefunction on a 512 x 512 grid, enough for first impressions
        Ψ,x,y=wavefunction_multi(solver,result.values[inds],[copy(result.vectors[i]) for i in inds],pts;ws=ws,nx_min=1024,ny_min=1024)
        @inbounds for (q,p) in enumerate(loc);Psi[p]=Ψ[q];end
        isnothing(xg)&&(xg=x;yg=y;pts0=pts)
    end
    plot_dielectric_wavefunctions(ks,Psi,xg,yg,pts0;maxcols=5) # 5 columns. Due to each contour having its own pts size we need to create wavefunction in a for loop for each contour
end
CairoMakie.save("compute_spectrum_wavef.png",fig)

################################################################################
################################ WEYL CHECK ####################################
################################################################################

# TM dielectric Weyl law in the odd-odd XY sector:
# ΔN≈A*n_in^2*(kmax^2-kmin^2)/(16π)+n_in*(rtilde*L0-LD)*(kmax-kmin)/(4π),
# with A=4aR+πR^2, L0=a+πR/2, LD=a+2R, and rtilde(1.5)≈1.025063134961447.
# Ref: E. Bogomolny, R. Dubertrand, C. Schmit, Phys. Rev. E 78, 056202 (2008),
# Eqs. (28), (29), and (63), inserted into (27).
stadium_a=0.5;stadium_R=1.0
A=4*stadium_a*stadium_R+pi*stadium_R^2 # it is because the a is the halfwidth of the staadium
Ldiel=stadium_a+pi*stadium_R/2
Lsym=stadium_a+2*stadium_R
rtilde=1.0250631349614472 # TM, n=1.5
Nweyl=A*n_in^2*(kmax^2-kmin^2)/(16*pi)+(rtilde*Ldiel-n_in*Lsym)*(kmax-kmin)/(4*pi)
Nnum=length(result.values)

@info "WEYL COUNT" numerical=Nnum weyl=Nweyl ratio=Nnum/Nweyl difference=Nnum-Nweyl