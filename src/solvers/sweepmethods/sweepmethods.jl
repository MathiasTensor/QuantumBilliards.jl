function solve_wavenumber(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners,CFIE_kress,CFIE_kress_corners,CFIE_kress_global_corners},basis::AbstractHankelBasis(),billiard::AbsBilliard,k,dk;multithreaded::Bool=true,use_krylov::Bool=false,which::Symbol=:det_argmin)
    pts=evaluate_points(solver,billiard,k)
    f(kk)=solve(solver,basis,pts,kk;multithreaded=multithreaded,use_krylov=use_krylov,which=which)
    res=Optim.optimize(f,k-dk/2,k+dk/2)
    return Optim.minimizer(res),Optim.minimum(res)
end