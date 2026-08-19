using QuantumBilliards
using LinearAlgebra
using Random
using Printf
using SpecialFunctions
using Statistics

try_MKL!()

# Validate Wiersig-Beyn resonances for a dielectric circle against the exact angular-momentum secular equation F_m(k)=0.

radius=1.0              # circle radius
n_in=1.5                # refractive index inside the cavity
n_out=1.0               # refractive index outside the cavity
ppw=10.0                # boundary points per wavelength
polarization=:TM        # dielectric polarization, TM vs TE
center=160.0-0.55im     # center of the rectangular Beyn contour
halfwidth=0.35          # contour half-width in Re(k)
halfheight=0.5          # contour half-height in Im(k)
nq=80                  # contour quadrature nodes
probe=400               # maximum random probe dimension
svd_tol=[1e-7,5e-8,1e-8,5e-9,1e-9,5e-10,1e-10] # SVD-rank thresholds
normalized_res_tol=1e-8 # normalized nonlinear residual acceptance threshold
validation_padding=5    # consecutive good candidates required by adaptive validation
circle_tol=1e-7         # maximum |F/F'| for agreement with the exact circle resonance
multithreaded=true      # threaded boundary-matrix assembly
dlp_kernel=:source      # source-normal DLP convention

circle,_=make_circle_and_basis(radius) # library geometry constructor
solver=WiersigKress(n_in,n_out,circle,ppw;quadrature_kind=:smooth,polarization=polarization) # Wiersig BIE solver. No corners so smooth trapezoidal quadraturate.
contour=wiersig_rectangle_contour(ComplexF64(center),halfwidth,halfheight) # a rectangle-ish contour that is smooth. Compromise between a good efficient tilling of the complex plane and being smooth for spectral convergence of teh contour integral.

# ANALYTIC CIRCLE HELPERS 

# Return J_m(z),J'_m(z),J''_m(z). The first derivative uses J'_m(z)=1/2[J_{m-1}(z)-J_{m+1}(z)],
# while J''_m follows from z²J''_m+zJ'_m+(z²-m²)J_m=0.
@inline function _jdata(m,z)
    J=besselj(m,z);Jp=(besselj(m-1,z)-besselj(m+1,z))/2
    return J,Jp,-Jp/z-(1-(m/z)^2)*J
end
# Same derivatives for the outgoing Hankel function H_m^(1).
@inline function _hdata(m,z)
    H=hankelh1(m,z);Hp=(hankelh1(m-1,z)-hankelh1(m+1,z))/2
    return H,Hp,-Hp/z-(1-(m/z)^2)*H
end
# Exact TM circle resonance equation:
# F_m=n_in J'_m(n_in Rk)H_m^(1)(n_out Rk)-n_out J_m(n_in Rk)H_m^(1)'(n_out Rk)=0.
# Its derivative is F'_m=R[n_in²J''_mH_m^(1)-n_out²J_mH_m^(1)''],
# since the mixed J'_mH'_m terms cancel. Thus |F/F'| is the local
# Newton estimate of the distance to the exact resonance.
@inline function circle_newton_error(k,m)
    x=n_in*radius*k;y=n_out*radius*k
    J,Jp,Jpp=_jdata(m,x);H,Hp,Hpp=_hdata(m,y)
    F=n_in*Jp*H-n_out*J*Hp
    Fp=radius*(n_in^2*Jpp*H-n_out^2*J*Hpp)
    return iszero(Fp) ? Inf : abs(F/Fp)
end
# Find the angular-momentum family with the smallest Newton correction.
# mmax≈n_in R Re(k)+20 safely covers the relevant channels.
function circle_error(k)
    δmin=Inf;mmin=0;mmax=ceil(Int,n_in*radius*abs(real(k))+20)
    @inbounds for m in 0:mmax
        δ=circle_newton_error(k,m)
        isfinite(δ)&&δ<δmin&&(δmin=δ;mmin=m)
    end
    return δmin,mmin
end
# Test all Beyn candidates against the exact circle equation.
function circle_truth(λ)
    δ=Vector{Float64}(undef,length(λ));m=Vector{Int}(undef,length(λ))
    Threads.@threads for j in eachindex(λ)
        δ[j],m[j]=circle_error(λ[j])
    end
    return (error=δ,m=m,bad=BitVector(δ .>=circle_tol))
end

# This is just one Beyn contour, not the typical compute_spectrum call. We dont need it here!
kres=hypot(abs(real(center))+halfwidth,abs(imag(center))+halfheight) # Build the contour-local discretization at the largest |k| on the contour.
pts=evaluate_points(solver,[max(n_in,n_out)*kres]) # generate the pts 
ws=build_cfie_kress_workspace(solver,pts) # make the geometry workspace and Kress params
r=min(probe,boundary_matrix_size(ws)) # in case user puts prove larger than the actual Fredholm matrix. Should never happen as probe space cant be larger than the nystrom matrix.

println("BEYN CIRCLE TEST: No chebyshev here since it must work on small RAM machines. But as a test reasonable speed")
println("center                 = ",center)
println("halfwidth / halfheight = ",halfwidth," / ",halfheight)
println("nq                     = ",nq)
println("probe                  = ",r)
println("SVD tolerances         = ",svd_tol)
println("padding                = ",validation_padding)
println()

tbeyn=@elapsed result=wiersig_beyn(solver,pts,ws,contour;nq=nq,r=r,r_step=r,max_r=r,svd_tol=svd_tol,relative_svd_tol=true,validate_roots=false,adaptive_validation=true,validation_padding=validation_padding,normalized_res_tol=normalized_res_tol,dlp_kernel=dlp_kernel,rng=MersenneTwister(0),multithreaded=multithreaded,verbose=true)

# Compare every enclosed Beyn candidate with the exact circle resonance equation.

λ=result.all_values;idx=findall(result.inside);truth=circle_truth(ComplexF64.(λ[idx]))
bad_global=idx[findall(truth.bad)]
unchecked_bad=[j for j in bad_global if !result.all_checked[j]]
kept_bad=[j for j in bad_global if result.kept[j]]

σ=result.all_effective_singular_values
imap=Dict(j=>q for (q,j) in enumerate(idx))
σorder=idx[sortperm(@view σ[idx])]
korder=sort(idx;by=j->(real(λ[j]),imag(λ[j])))

@info "σeff ORDER"
println(rpad("p",5),rpad("j",6),rpad("σeff",14),rpad("Re(k)",15),rpad("Im(k)",15),rpad("m",6),rpad("δk circle",14),rpad("truth",8),rpad("checked",9),"residual")

@inbounds for (p,j) in enumerate(σorder)
    q=imap[j];nr=result.all_checked[j] ? @sprintf("%.3e",result.all_normalized_residuals[j]) : "—"
    println(rpad(p,5),rpad(j,6),rpad(@sprintf("%.3e",σ[j]),14),rpad(@sprintf("%.8f",real(λ[j])),15),rpad(@sprintf("%.8f",imag(λ[j])),15),rpad(truth.m[q],6),rpad(@sprintf("%.3e",truth.error[q]),14),rpad(truth.bad[q] ? "BAD" : "GOOD",8),rpad(result.all_checked[j],9),nr)
end

@info "k ORDER"
println(rpad("p",5),rpad("j",6),rpad("Re(k)",15),rpad("Im(k)",15),rpad("|Δk prev|",14),rpad("σeff",14),rpad("m",6),rpad("δk circle",14),rpad("truth",8),rpad("checked",9),"residual")

@inbounds for (p,j) in enumerate(korder)
    q=imap[j];Δ=p==1 ? NaN : abs(λ[j]-λ[korder[p-1]]);nr=result.all_checked[j] ? @sprintf("%.3e",result.all_normalized_residuals[j]) : "—"
    println(rpad(p,5),rpad(j,6),rpad(@sprintf("%.8f",real(λ[j])),15),rpad(@sprintf("%.8f",imag(λ[j])),15),rpad(p==1 ? "—" : @sprintf("%.3e",Δ),14),rpad(@sprintf("%.3e",σ[j]),14),rpad(truth.m[q],6),rpad(@sprintf("%.3e",truth.error[q]),14),rpad(truth.bad[q] ? "BAD" : "GOOD",8),rpad(result.all_checked[j],9),nr)
end

isempty(unchecked_bad)&&isempty(kept_bad) ? @info("Circle validation passed") : begin
    !isempty(unchecked_bad)&&@warn "Unchecked false circle candidates" indices=unchecked_bad
    !isempty(kept_bad)&&@warn "False circle candidates retained" indices=kept_bad
end

# This highlights that a few spurious improve when we increase rank. So a small number of directions with low SV can asbolutely increase the accuracy of roots. This is quite robust as we can remove the older less converged roots with those using a lower svd_tol with a larger rank. And ofc those with really small σeff are automatically spurios as per Beyn's paper
#= EXAMPLE:
p    j     σeff          Re(k)          Im(k)          m     δk circle     truth   checked  residual
1    206   3.479e-06     160.30543563   -0.49875396    84    2.124e-02     BAD     true     4.166e-04
2    205   2.823e-04     159.73529022   -0.49926960    61    2.151e-02     BAD     true     5.478e-04
# the first 2 are way in the numerical nullspace. They correspind to no circle solution and their residual is large.
3    93    1.466e+02     159.94182867   -0.05956187    163   6.439e-13     GOOD    true     6.864e-10
4    94    1.475e+02     159.94182867   -0.05956187    163   7.553e-13     GOOD    true     7.335e-10
=#