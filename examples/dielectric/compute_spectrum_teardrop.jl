using QuantumBilliards
using LinearAlgebra
using StaticArrays
using Random
using Printf
using Statistics
using CairoMakie

try_MKL!()

n_in=1.5 # interior refractive index
n_out=1.0 # exterior refractive index
ppw=10.0 # boundary points per wavelength
polarization=:TM # dielectric polarization
kmin=45.0 # lower Re(k)
kmax=50.0 # upper Re(k)
interval_width=5.0 # real-k resolution interval width
im_min=-0.9 # lower Im(k)
im_max=-0.01 # upper Im(k)
halfwidth=0.4 # rectangular contour half-width
halfheight=0.5 # rectangular contour half-height
nx_GL=18 # horizontal-edge Gauss-Legendre order
ny_GL=18 # vertical-edge Gauss-Legendre order
svd_tol=collect(1e-6*0.1.^(0:4)) # absolute SVD-rank thresholds
normalized_res_tol=1e-11 # normalized nonlinear residual threshold
validation_padding=5 # consecutive passes after the last linear-scan failure
adaptive_validation_method=:binary # fast adaptive residual validator (try :linear for slower but more robust)
multithreaded=true # threaded matrix construction
dlp_kernel=:source # source-normal DLP convention for wavefucntions plotting
chebyshev=false # Chebyshev-accelerated matrix assembly (if too large k RAM might explode on old machines)
verbose=true # print solver diagnostics
print_spectrum=false # print individual resonances

billiard=TeardropBilliard() # full teardrop cavity
symmetry=YReflection(-1) # odd under reflection across the y axis, x -> -x
solver=WiersigKress(n_in,n_out,billiard,ppw;quadrature_kind=:corners,polarization=polarization,symmetry=symmetry,kressq=2) # corner-graded Kress quadrature at the cusp

template=WiersigRectangleContour(ComplexF64(0,(im_min+im_max)/2),halfwidth,halfheight) # rectangular Beyn contour template
kwargs=(nq=(nx_GL,ny_GL),svd_tol=svd_tol,relative_svd_tol=false,validate_roots=false,adaptive_validation=true,adaptive_validation_method=adaptive_validation_method,validation_padding=validation_padding,normalized_res_tol=normalized_res_tol,dlp_kernel=dlp_kernel,multithreaded=multithreaded,chebyshev=chebyshev,verbose=verbose,probe_factor=5.0) # production Beyn options

@info "COMPUTE_SPECTRUM TEST" geometry=:Teardrop symmetry=symmetry region=(kmin,kmax,im_min,im_max) nq=(nx_GL,ny_GL)
tbeyn=@elapsed result=compute_spectrum(solver,kmin,kmax,im_min,im_max,template;interval_width=interval_width,kwargs...) # compute resonances
@info "SPECTRUM SUMMARY" time=round(tbeyn,digits=3) roots=length(result.values) intervals=length(result.interval_results) contours=length(result.contours) Nmin=minimum(result.contour_dimensions) Nmax=maximum(result.contour_dimensions)

perm=sortperm(eachindex(result.values);by=i->(real(result.values[i]),imag(result.values[i]))) # sort roots for output and plotting

if print_spectrum
    println(rpad("i",6),rpad("Re(k)",17),rpad("Im(k)",17),rpad("residual",14),"contour")
    for (p,i) in enumerate(perm)
        k=result.values[i];nr=result.normalized_residuals[i]
        println(rpad(p,6),rpad(@sprintf("%.12f",real(k)),17),rpad(@sprintf("%.12f",imag(k)),17),rpad(isfinite(nr) ? @sprintf("%.3e",nr) : "—",14),result.source_contours[i])
    end
end
# plot min (50,number of roots) wavefunctions on a Cartesian grid
if !isempty(perm)
    nplot=min(50,length(perm));sel=perm[end-nplot+1:end] 
    ks=result.values[sel];Psi=Vector{Matrix{ComplexF64}}(undef,nplot)
    fig=let xg=nothing,yg=nothing,pts0=nothing
        for ic in unique(result.source_contours[sel])
            loc=findall(p->result.source_contours[sel[p]]==ic,eachindex(sel))
            inds=sel[loc];pts=result.contour_pts[ic];ws=result.contour_workspaces[ic]
            Ψ,x,y=wavefunction_multi(solver,result.values[inds],[copy(result.vectors[i]) for i in inds],pts;ws=ws,nx_min=1024,ny_min=1024) # reconstruct modes on a Cartesian grid
            @inbounds for (q,p) in enumerate(loc);Psi[p]=Ψ[q];end
            isnothing(xg)&&(xg=x;yg=y;pts0=pts)
        end
        plot_dielectric_wavefunctions(ks,Psi,xg,yg,pts0;maxcols=5)
    end
    CairoMakie.save("teardrop_compute_spectrum_wavef.png",fig)
end