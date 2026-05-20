using QuantumBilliards
using LinearAlgebra
using CairoMakie
using Statistics

billiard=make_rectangle_and_basis(1.1,1.0)[1]
bs=collect(5.0:1.0:15.0) # bs from 6.0 to 15.0 for k1=200.0 to k2=220.0
symmetry=XYReflection(-1,-1)

k1=200.0
k2=220.0

ks_per_b=Dict{Float64,Vector{ComplexF64}}() # dictionary to store computed eigenvalues for each b
im_ks_per_b=Dict{Float64,Vector{Float64}}() # dictionary to store computed imaginary parts for each b
errs_per_b=Dict{Float64,Vector{Float64}}() # dictionary to store errors for each b to compare to im part

for b in bs
solver_rcip=DLP_rcip(b,billiard,nsub=40,ngl=16,symmetry=symmetry,min_pts=15)

@time "Beyn" ks,tens,us,pts,tensN=compute_spectrum_beyn(
    solver_rcip,                     # solver defining the boundary operator T(k)
    billiard,                        # billiard used for Weyl planning and point evaluation
    k1,                              # lower scan bound
    k2;                              # upper scan bound
    m=100,                           # target number of levels per planned window
    Rmax=0.5,                        # cap on contour radius
    nq=40,                           # number of contour nodes per disk
    r=150,                           # Beyn probe rank / number of random test vectors
    svd_tol=1e-12,                   # SVD rank-detection threshold
    res_tol=1e-9,                    # residual tolerance for filtering roots
    auto_discard_spurious=true,      # whether to remove large-residual roots
    multithreaded_matrix=true,       # matrix assembly threading flag
    use_chebyshev=true,              # use Chebyshev Hankel interpolation for complex contour evals (small problem, but still currently only supported path)
    return_imag_part=true,           # whether to return the imaginary part of the spectrum for quad error estimate
    use_imag_check_EXPERIMENTAL=true # faster way to check for spurious roots 
)

# store results in dictionaries for later analysis
ks_per_b[b]=ks
im_ks_per_b[b]=imag.(ks)
end

# odd-odd rectangle analytical eigenvalues
function rect_exact_xyodd(a,b)
    ks=Float64[]
    mmax=ceil(Int,k2*a/(2*pi))+5
    nmax=ceil(Int,k2*b/(2*pi))+5
    for m in 1:mmax
        for n in 1:nmax
            k=2*pi*sqrt((m/a)^2+(n/b)^2)
            k1<=k<=k2 && push!(ks,k)
        end
    end
    sort!(unique(ks))
    return ks
end

# match numerical and exact spectra by closest distance, ensuring each numerical eigenvalue is matched at most once
function match_spectra(k_exact,k_num)
    matched=Tuple{Float64,Float64,Float64}[]
    used=falses(length(k_num))
    for ke in k_exact
        best_i=0
        best_err=Inf
        for (i,kn) in enumerate(k_num)
            used[i] && continue
            err=abs(kn-ke)
            if err<best_err
                best_err=err
                best_i=i
            end
        end
        if best_i!=0
            push!(matched,(ke,k_num[best_i],best_err))
            used[best_i]=true
        end
    end
    return matched
end

k_exact=rect_exact_xyodd(1.1,1.0)

# analyze errors for each b by matching numerical eigenvalues to exact ones
for b in bs
k_num=sort(real.(ks_per_b[b]))
matched=match_spectra(k_exact,k_num)
errs=getindex.(matched,3)
errs_per_b[b]=errs
end

med_errs=[median(abs.(errs_per_b[b])) for b in bs]
med_ims=[median(abs.(im_ks_per_b[b])) for b in bs]
fig=Figure(size=(1000,700))
ax1=Axis(fig[1,1],xlabel=L"b",ylabel=L"\mathrm{median}\ ||k_{\mathrm{num}}-k_{\mathrm{exact}}|",yscale=log10,ylabelcolor=:blue,yticklabelcolor=:blue,leftspinecolor=:blue,xlabelsize=30,ylabelsize=30,xticksize=25,yticksize=25,backgroundcolor=(:white,0.0),xgridvisible=false,ygridvisible=false)
scatterlines!(ax1,bs,med_errs;color=:blue,marker=:diamond,markersize=14,linewidth=3,label=L"\mathrm{median}\ ||k_{\mathrm{num}}-k_{\mathrm{exact}}|")
ax2=Axis(fig[1,1],yaxisposition=:right,yscale=log10,ylabel=L"\mathrm{median}\ ||\Im\,k|",ylabelcolor=:red,yticklabelcolor=:red,rightspinecolor=:red,xticklabelsvisible=false,xticksvisible=false,xgridvisible=false,xlabelsize=30,ylabelsize=30,xticksize=25,yticksize=25,backgroundcolor=(:white,0.0),ygridvisible=false)
scatterlines!(ax2,bs,med_ims;color=:red,marker=:circle,markersize=14,linewidth=3,label=L"\mathrm{median}\ ||\Im\,k|")
axislegend(ax1,position=:rt)
axislegend(ax2,position=:lb)
save("rcip_dual_axis_convergence.pdf",fig)
save("rcip_dual_axis_convergence.png",fig)