using QuantumBilliards
using LinearAlgebra
using CairoMakie
using CSV
using DataFrames
using Statistics

try_MKL!()

billiard,basis=make_ellipse_and_basis(1.0,0.5)

d=8.0
b=12.0

k1=5.0
k2=400.0

symmetry=XYReflection(-1,-1) 
#solver_dlp_kress=DLP_kress(b,billiard,symmetry=symmetry) # very slow as it needs to do whole matrix, but extremely accurate - imag beyn eigval proxy for accuracy - 1e-14 up to 1e-15 im part!
solver_dlp=BoundaryIntegralMethod(b,billiard,symmetry=symmetry) # faster than Kress but less accurate, still should give enough digits for good spectral statistics < 1 % of mean level spacing, 5 digits is enough for this (that is the im part)
solver_vs=VerginiSaraceno(d,b)

which_solver=:vs # or :beyn

if which_solver==:beyn # (benchmark Beyn vs Vergini Saraceno, really similar performance if matrix sizes are equal)

!isfile("ellipse_ks_beyn.csv") && begin

# solving with dlp_kress is 4x more difficult problem as it needs to solve full matrix before projecting to irrep subspace
# so here we just do standard BoundaryIntegralMethod for Beyn, which should be enough to get good accuracy for spectral statistics while allowing for matrix size reduction from symmetry.
@time "Beyn" ks,tens,us,pts,tensN=compute_spectrum_beyn(
    solver_dlp,                      # solver defining the boundary operator T(k)
    billiard,                        # billiard used for Weyl planning and point evaluation
    k1,                              # lower scan bound
    k2;                              # upper scan bound
    m=100,                           # target number of levels per planned window
    Rmax=0.7,                        # cap on contour radius
    nq=45,                           # number of contour nodes per disk
    r=150,                           # Beyn probe rank / number of random test vectors
    svd_tol=1e-12,                   # SVD rank-detection threshold
    res_tol=1e-9,                    # residual tolerance for filtering roots
    auto_discard_spurious=true,      # whether to remove large-residual roots
    multithreaded_matrix=true,       # matrix assembly threading flag
    use_chebyshev=true,              # use Chebyshev Hankel interpolation for complex contour evals (small problem, but still currently only supported path)
)

CSV.write("ellipse_ks_beyn.csv",DataFrame(k=ks))

end

end

if which_solver==:vs

!isfile("ellipse_ks_vs.csv") && begin 

# Vergini Saraceno second (this is solving the symmetry reduced problem from the start, so should be 4x easier than Beyn)
dk=0.05
@time "VerginiSaraceno" state_res,_=compute_spectrum_with_state_scaling_method(
    solver_vs,
    basis, # the RPW or CAFB basis (from make_ellipse_and_basis), with corner angle pi/2 at the origin
    billiard,
    k1,
    k2,
    dk, # heuristic dk interval, should not really be a constant, actually spectrum should be computed in chunks and checked if levels are missing in any
    multithreaded_matrices=true, # just leave on always, matrices should always be multithreaded for assembly
    multithreaded_ks=false, # for large problems this is best left to false to avoid oversubscription
)

ks=state_res.ks
CSV.write("ellipse_ks_vs.csv",DataFrame(k=ks))

end

end

# load the data
if which_solver==:beyn
    df=CSV.read("ellipse_ks_beyn.csv",DataFrame)
    ks=df.k
elseif which_solver==:vs
    df=CSV.read("ellipse_ks_vs.csv",DataFrame)
    ks=df.k
end

# Plot P(s) after unfolding the spectrum with Weyl's law
ks=weyl_law(collect(ks),billiard,fundamental=true) # we are using eigvals from a fundamental domain
spacings=calculate_spacings(ks)
@info "mean spacing: $(mean(spacings)) (should be 1 after unfolding)"

# generate histogram of spacings and compare with Poisson
function spacing_histogram(s;nbins=40)
    n=length(s)
    smin=minimum(s)
    smax=maximum(s)
    # bin edges
    edges=range(smin,smax;length=nbins+1)
    widths=diff(edges)
    counts=zeros(Float64,nbins)
    # fill histogram
    for x in s
        if x==smax
            idx=nbins
        else
            idx=Int(floor((x-smin)/(smax-smin)*nbins))+1
            idx=clamp(idx,1,nbins)
        end
        counts[idx]+=1
    end
    # sum(pdf*bin_width)=1
    total=0.0
    for i in 1:nbins
        total+=counts[i]*widths[i]
    end
    pdf=counts./total
    # bin centers
    centers=[(edges[i]+edges[i+1])/2 for i in 1:nbins]
    return centers, pdf
end

centers,pdf=spacing_histogram(spacings,nbins=30)
f=Figure()
ax=Axis(f[1,1],xlabel="s",ylabel="P(s)")
barplot!(ax,centers,pdf,color=:blue,bar_width=0.9*(centers[2]-centers[1]))
s_poisson=range(0.0,5.0;length=1000)
P_poisson=exp.(-s_poisson)
lines!(ax,s_poisson,P_poisson,color=:red,label="Poisson")
axislegend(ax)
save("spacing_distribution_$(which_solver).png",f)