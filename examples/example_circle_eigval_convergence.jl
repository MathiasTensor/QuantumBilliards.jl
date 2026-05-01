using QuantumBilliards
using CairoMakie
using Statistics

# Check convergence of different BIE solvers towards a circle eigenvalue
# cirle has analytic solutions and no corners, so we excpect that 
# DLP_kress and CFIE_kress should give exponential convergence, 
# Alpert should give high order algebraic convergence (14-16 for the used rules in this example
# and BIM should give 3rd order algebraic convergence since it is weakly singular as
# we dont fix the near diagonal behaviouur.

try_MKL!()

# exact eigenvalue (J_0 zero near k ≈ 30)
k_exact=30.63460646843198
# small sweep window, should start close to an eigenvalue
k1=k_exact-0.05
k2=k_exact+0.05
dk=2e-4
kgrid=collect(k1:dk:k2)

# resolution parameters
bvals=[2.0,3.0,4.0,5.0,6.0,8.0,10.0] # these are points per wavelength, so the number of points N = b*k/(2*pi) for the circle

# geometry
billiard,_=make_circle_and_basis(1.0)

# storage
names=["DLP","DLP_kress","CFIE_kress","CFIE_alpert (order=12, q=4)","CFIE_alpert (order=14, q=2)"]
kconv=Dict(name=>Float64[] for name in names)
errconv=Dict(name=>Float64[] for name in names)
Nconv=Dict(name=>Int[] for name in names)

for b in bvals
    println("b = $b")
    solvers=[
        ("DLP",BoundaryIntegralMethod(b,billiard)),
        ("DLP_kress",DLP_kress(b,billiard)),
        ("CFIE_kress",CFIE_kress(b,billiard)),
        ("CFIE_alpert (order=12, q=4)",CFIE_alpert(b,billiard,alpert_order=12,alpertq=4)),
        ("CFIE_alpert (order=14, q=2)",CFIE_alpert(b,billiard,alpert_order=14,alpertq=2)),
    ]
    for (name,solver) in solvers
        tens=k_sweep(solver,AbstractHankelBasis(),billiard,kgrid)
        kmins,tmins=refine_minima(solver,AbstractHankelBasis(),billiard,kgrid,tens,pts_refinement_factors=(1.0),dim_refinement_factors=(1.0),print_refinement=false) # no increase in b (pts_refinement_factors=(1.0) -> 
        # this means b will not be increased in the refinement giving true minima for that b)
        # we want to know for a given b what is the accuracy of the closest minimum to the exact k
        # print_refinement=false makes it so the terminal output is not cluttered
        idx=argmin(abs.(kmins.-k_exact))
        k_num=kmins[idx]
        push!(kconv[name],k_num)
        push!(errconv[name],abs(k_num-k_exact))
        push!(Nconv[name],round(Int,k_exact*b))
        println("  $name -> err = $(abs(k_num-k_exact))")
    end
end

##### BIM FIT #####

# extract BIM data
N_bim=Nconv["DLP"]
err_bim=errconv["DLP"].+1e-16
# fit constant C using first point and ref curve
C=err_bim[1]*N_bim[1]^3 # we expect BIM to have 3rd order convergence since it is weakly singular and we dont fix the near diagonal behaviour, so the reference curve is C/N^3
Ns=range(minimum(N_bim),maximum(N_bim),length=200)
ref=C./(Ns.^3)

##### ALPERT FIT #####

# estimate convergence rate for CFIE_alpert
N_alp=Nconv["CFIE_alpert (order=12, q=4)"]
err_alp=errconv["CFIE_alpert (order=12, q=4)"].+1e-16

# avoid machine precision plateau (keep meaningful points)
mask=err_alp.>1e-13
x=log.(N_alp[mask])
y=log.(err_alp[mask])

# linear regression slope
mx=mean(x);my=mean(y)
slope=sum((x.-mx).*(y.-my))/sum((x.-mx).^2)
p_est=-slope # slope for the used Alpert rules should be 12-14
# fit prefactor for Alpert
C_alp=exp(mean(log.(err_alp[mask].*(N_alp[mask].^p_est))))
ref_alp=C_alp./(Ns.^p_est)

# plot everything
fig=Figure()
ax=Axis(fig[1,1],xlabel=L"N",ylabel=L"|k - k_{exact}|",yscale=log10,xscale=log10,title="Circle R=1 convergence k = 30.63460646843198")

for name in names
    lines!(ax,Nconv[name],errconv[name].+1e-16,label=name)
    scatter!(ax,Nconv[name],errconv[name].+1e-16)
end

# add 1/N^3 reference
lines!(ax,Ns,ref,linestyle=:dash,color=:black,label="~ N^{-3}")
# add Alpert reference
lines!(ax,Ns,ref_alp,linestyle=:dash,color=:red,label="~ N^{-$(round(p_est,digits=2))}")
axislegend(ax,position=:rc)
save("circle_convergence.png",fig)