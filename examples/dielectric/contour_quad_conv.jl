using QuantumBilliards
using LinearAlgebra
using Random
using Printf
using Statistics
using CairoMakie

try_MKL!()

#NOTE: This is a very long test because it discretizes a contour many times with different quadrature orders. It is not meant to be run frequently, but rather to be used for a final convergence check of the Beyn contour-quadrature implementation.
# For a realistic run just use the wanted qudrature order and some below it and test teh accuracy and number of solutions
# if the quadrature is resolved enough.
#NOTE: GL is cheaper even if time in the table to std::out shows +Xs or more diff, because this is jsut one contour. For the full spectrum it reuses common edges of cells and can save a lot of time.

# =============================================================================
# BEYN CONTOUR-QUADRATURE CONVERGENCE TEST
# The Wiersig boundary discretization and random probe are held fixed.
# Only the Beyn contour quadrature is changed.
# Smooth periodic contour: Nsolve=nq
# Exact rectangle with (n,n) Gauss-Legendre: Nsolve=4n
# fit mean|Δk| ~ C exp(-α Nsolve)
# =============================================================================

n_in=1.5
n_out=1.0
ppw=10.0
polarization=:TM
center=30.5-0.45im
halfwidth=0.40
halfheight=0.40
smooth_orders=[24,28,32,40,48,56,64,72,80] # periodic-trapezoidal node counts used for the smooth-contour convergence sweep
smooth_reference_nq=160                         # high-order smooth-contour quadrature used to define the reference resonance set
rectangle_orders=[6,7,8,10,12,14,16,18,20,24] # Gauss-Legendre order per rectangle edge used for the convergence sweep
rectangle_reference=(40,40)                    # high-order horizontal/vertical Gauss-Legendre orders used for the rectangle reference
probe=400                                      # requested Beyn probe dimension, clipped to the boundary-matrix dimension when necessary
svd_tol=collect(1e-8*0.1.^(0:5))               # decreasing absolute SVD thresholds used to generate the candidate numerical-rank ladder
#TODO IMPORTANT: normalized_res_tol is basically a proxy for accuracy (see results after running this file)
normalized_res_tol=1e-13                       # normalized residual required for an accepted resonance
validation_padding=5                           # number of consecutive passing candidates required before adaptive residual checking
relative_svd_tol=false                         # interpret `svd_tol` as absolute singular-value thresholds rather than relative to σ₁
validation_padding=5
relative_svd_tol=false
multithreaded=true
dlp_kernel=:source
verbose=false
# Root matching cutoff, large b/c low-order runs can initially be substantially displaced.
match_tol=0.10
# Do not include the numerical plateau in the exponential fit.
fit_floor=1e-13

################################################################################
################################ GEOMETRY #######################################
################################################################################

billiard,_=make_stadium_and_basis(0.5)
symmetry=XYReflection(-1,-1)
solver=WiersigKress(n_in,n_out,billiard,ppw;quadrature_kind=:global_corners,polarization=polarization,symmetry=symmetry)
smooth_contour=wiersig_fourier_rectangle_contour(ComplexF64(center),halfwidth,halfheight)
rectangle_contour=WiersigRectangleContour(ComplexF64(center),halfwidth,halfheight)
kres=hypot(abs(real(center))+halfwidth,abs(imag(center))+halfheight)
pts=evaluate_points(solver,[max(n_in,n_out)*kres])
ws=build_cfie_kress_workspace(solver,pts)
r=min(probe,boundary_matrix_size(ws))

@info "CONVERGENCE TEST" geometry=nameof(typeof(billiard)) symmetry=symmetry center=center halfwidth=halfwidth halfheight=halfheight N=boundary_matrix_size(ws) probe=r

################################################################################
############################## ROOT MATCHING ####################################
################################################################################

function match_roots(k::AbstractVector,kref::AbstractVector;tol=Inf)
    isempty(k)&&return (errors=Float64[],i=Int[],j=Int[],unmatched=0)
    isempty(kref)&&return (errors=Float64[],i=Int[],j=Int[],unmatched=length(k))
    pairs=Vector{Tuple{Float64,Int,Int}}(undef,length(k)*length(kref));p=0
    @inbounds for i in eachindex(k),j in eachindex(kref)
        p+=1;pairs[p]=(abs(k[i]-kref[j]),i,j)
    end
    sort!(pairs;by=first)
    used=falses(length(k));usedref=falses(length(kref));errs=Float64[];ii=Int[];jj=Int[]
    @inbounds for (d,i,j) in pairs
        d>tol&&break
        if !used[i]&&!usedref[j]
            used[i]=true;usedref[j]=true
            push!(errs,d);push!(ii,i);push!(jj,j)
        end
    end
    return (errors=errs,i=ii,j=jj,unmatched=count(!,used))
end

################################################################################
############################ SINGLE CONTOUR SOLVE ##############################
################################################################################

function solve_contour(contour,nq)
    t=@elapsed result=wiersig_beyn(solver,pts,ws,contour;nq=nq,r=r,r_step=r,max_r=r,svd_tol=svd_tol,relative_svd_tol=relative_svd_tol,validate_roots=false,adaptive_validation=true,validation_padding=validation_padding,normalized_res_tol=normalized_res_tol,dlp_kernel=dlp_kernel,rng=MersenneTwister(0),multithreaded=multithreaded,verbose=verbose)
    return result,t
end

################################################################################
################################ REFERENCES #####################################
################################################################################

println()
@info "SMOOTH REFERENCE" nq=smooth_reference_nq
smooth_ref,tsref=solve_contour(smooth_contour,smooth_reference_nq)

@info "RECTANGLE REFERENCE" nq=rectangle_reference
rectangle_ref,trref=solve_contour(rectangle_contour,rectangle_reference)

ksref=smooth_ref.values
krref=rectangle_ref.values

@info "REFERENCE SUMMARY" smooth_roots=length(ksref) smooth_time=round(tsref,digits=3) rectangle_roots=length(krref) rectangle_time=round(trref,digits=3)

################################################################################
############################ SMOOTH CONVERGENCE ################################
################################################################################

smooth_mean=fill(NaN,length(smooth_orders))
smooth_median=fill(NaN,length(smooth_orders))
smooth_max=fill(NaN,length(smooth_orders))
smooth_matched=zeros(Int,length(smooth_orders))
smooth_roots=zeros(Int,length(smooth_orders))
smooth_time=zeros(Float64,length(smooth_orders))

println()
println("SMOOTH FOURIER CONTOUR")
println(rpad("nq",8),rpad("nodes",10),rpad("roots",8),rpad("rank",8),rpad("svdtol",12),rpad("ranks",24),rpad("matched",10),rpad("mean |Δk|",16),rpad("median |Δk|",16),rpad("max |Δk|",16),"time [s]")

for (q,nq) in enumerate(smooth_orders)
    R,t=solve_contour(smooth_contour,nq);M=match_roots(R.values,ksref;tol=match_tol);e=M.errors
    smooth_roots[q]=length(R.values);smooth_matched[q]=length(e);smooth_time[q]=t
    if !isempty(e);smooth_mean[q]=mean(e);smooth_median[q]=median(e);smooth_max[q]=maximum(e);end
    ranks_str=string(R.ranks)
    println(rpad(nq,8),rpad(nq,10),rpad(smooth_roots[q],8),rpad(R.rank,8),rpad(@sprintf("%.1e",R.selected_svd_tolerance),12),rpad(ranks_str,24),rpad(smooth_matched[q],10),rpad(isfinite(smooth_mean[q]) ? @sprintf("%.6e",smooth_mean[q]) : "—",16),rpad(isfinite(smooth_median[q]) ? @sprintf("%.6e",smooth_median[q]) : "—",16),rpad(isfinite(smooth_max[q]) ? @sprintf("%.6e",smooth_max[q]) : "—",16),@sprintf("%.3f",t))
end

################################################################################
########################## RECTANGLE CONVERGENCE ###############################
################################################################################

rectangle_mean=fill(NaN,length(rectangle_orders))
rectangle_median=fill(NaN,length(rectangle_orders))
rectangle_max=fill(NaN,length(rectangle_orders))
rectangle_matched=zeros(Int,length(rectangle_orders))
rectangle_roots=zeros(Int,length(rectangle_orders))
rectangle_time=zeros(Float64,length(rectangle_orders))

println()
println("RECTANGLE — GAUSS-LEGENDRE")
println(rpad("n",8),rpad("nodes",10),rpad("roots",8),rpad("rank",8),rpad("svdtol",12),rpad("ranks",24),rpad("matched",10),rpad("mean |Δk|",16),rpad("median |Δk|",16),rpad("max |Δk|",16),"time [s]")

for (q,n) in enumerate(rectangle_orders)
    R,t=solve_contour(rectangle_contour,(n,n));M=match_roots(R.values,krref;tol=match_tol);e=M.errors
    rectangle_roots[q]=length(R.values);rectangle_matched[q]=length(e);rectangle_time[q]=t
    if !isempty(e);rectangle_mean[q]=mean(e);rectangle_median[q]=median(e);rectangle_max[q]=maximum(e);end
    ranks_str=string(R.ranks)
    println(rpad(n,8),rpad(4n,10),rpad(rectangle_roots[q],8),rpad(R.rank,8),rpad(@sprintf("%.1e",R.selected_svd_tolerance),12),rpad(ranks_str,24),rpad(rectangle_matched[q],10),rpad(isfinite(rectangle_mean[q]) ? @sprintf("%.6e",rectangle_mean[q]) : "—",16),rpad(isfinite(rectangle_median[q]) ? @sprintf("%.6e",rectangle_median[q]) : "—",16),rpad(isfinite(rectangle_max[q]) ? @sprintf("%.6e",rectangle_max[q]) : "—",16),@sprintf("%.3f",t))
end

################################################################################
############################ EXPONENTIAL FITS ##################################
################################################################################

#=

#NOTE: Only if the normalized_res_tol is sufficiently large can one see the exponential convergence nicely
# otherwise typically at a minimal numbe of quadrature nodes it will just converge to the wanted accuracy

# Fit log(error)=log(C)-α Nsolve.
# Points below `floor` are omitted because they are already on the numerical
# accuracy plateau rather than in the contour-quadrature convergence regime.
function exponential_fit(n,e;floor=0.0)
    idx=findall(i->isfinite(e[i])&&e[i]>floor,eachindex(e))
    length(idx)>=2||return (C=NaN,alpha=NaN,fit=fill(NaN,length(n)),idx=idx)
    x=Float64[n[i] for i in idx];y=log.(Float64[e[i] for i in idx])
    β=hcat(ones(length(x)),x)\y
    C=exp(β[1]);α=-β[2]
    return (C=C,alpha=α,fit=C.*exp.(-α.*Float64.(n)),idx=idx)
end

smooth_nodes=Float64.(smooth_orders)
rectangle_nodes=4.0.*Float64.(rectangle_orders)
fs=exponential_fit(smooth_nodes,smooth_mean;floor=fit_floor)
fr=exponential_fit(rectangle_nodes,rectangle_mean;floor=fit_floor)
println()
@info "EXPONENTIAL CONVERGENCE FIT PER MATRIX SOLVE" fit_floor=fit_floor smooth_alpha=fs.alpha smooth_C=fs.C smooth_points=fs.idx rectangle_alpha=fr.alpha rectangle_C=fr.C rectangle_points=fr.idx
println("smooth:    mean|Δk| ≈ ",@sprintf("%.4e",fs.C)," exp(-",@sprintf("%.6f",fs.alpha)," Nsolve)")
println("rectangle: mean|Δk| ≈ ",@sprintf("%.4e",fr.C)," exp(-",@sprintf("%.6f",fr.alpha)," Nsolve)")

fig=Figure(size=(950,680))
ax=Axis(fig[1,1],xlabel="number of contour matrix solves",ylabel="mean |Δk|",yscale=log10,title="Beyn contour-quadrature convergence")
scatter!(ax,smooth_nodes,smooth_mean;label="smooth trapezoidal")
lines!(ax,smooth_nodes,fs.fit;label="smooth fit: α=$(@sprintf("%.3f",fs.alpha))")
scatter!(ax,rectangle_nodes,rectangle_mean;label="rectangle Gauss-Legendre")
lines!(ax,rectangle_nodes,fr.fit;label="rectangle fit: α=$(@sprintf("%.3f",fr.alpha))")
xmin=min(minimum(smooth_nodes),minimum(rectangle_nodes))
xmax=max(maximum(smooth_nodes),maximum(rectangle_nodes))
lines!(ax,[xmin,xmax],[fit_floor,fit_floor];linestyle=:dash,label="fit floor")
axislegend(ax;position=:rt)
save("contour_convergence.png",fig)

=#