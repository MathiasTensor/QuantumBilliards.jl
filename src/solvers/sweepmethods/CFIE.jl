
##########################################################################################################
#### WEIGHT FUNCTIONS USED BY KRESS: Boundary Integral Equations in time-harmonic acoustic scattering ####
##########################################################################################################
# The parameter s ∈ [0,2π] in all non-reparametried functions. We rescale it since segment parametrizations go from [0,1]
v(s::T,q::Int) where {T<:Real}=(1/q-1/2)*((pi-s)/pi)^3+1/q*((s-pi)/pi)+0.5
dv(s::T,q::Int) where {T<:Real}=-(3*(1/q-1/2)/π)*((π-s)/π)^2+1/(q*π)
w_kress(s::T,q::Int) where {T<:Real}=2*pi*(v(s,q)^q)/(v(s,q)^q+v(2*pi-s,q)^q)
w_reparametrized(s::T,q::Int) where {T<:Real}=w_kress(2*pi*s,q)/(2*pi)
function dw_reparametrized(t::T,q::Int) where {T<:Real}
    s=2π*t
    As=v(s,q)^q
    Cs=v(2*pi-s,q)^q
    Bs=As+Cs
    dAs=q*v(s,q)^(q-1)*dv(s,q)
    dCs=-q*v(2π-s,q)^(q-1)*dv(2π-s,q)
    dwk=2π*(dAs*Bs-As*(dAs+dCs))/Bs^2
    return dwk
end

v(s::AbstractVector{T},q::Int) where {T<:Real}=v.(s,q)
dv(s::AbstractVector{T},q::Int) where {T<:Real}=dv.(s,q)
w_kress(s::AbstractVector{T},q::Int) where {T<:Real}=w_kress.(s,q)
w_reparametrized(s::AbstractVector{T},q::Int) where {T<:Real}=w_reparametrized.(s,q)
dw_reparametrized(s::AbstractVector{T},q::Int) where {T<:Real}=dw_reparametrized.(s,q)

###########################
#### CONSTRUCTOR CFIE ####
###########################

struct CFIE{T}<:SweepSolver where {T<:Real} 
    fundamental::Bool
    sampler::Vector{LinearNodes} # placeholder since the trapezoidal rule will be rescaled
    pts_scaling_factor::Vector{T}
    ws::Vector{Function} # quadrature weights for each segment, must be same length as the length of "fundamental::Bool" boundary, if true same as fundamental boundary, otherwise full boundary
    ws_der::Vector{Function} # quadrature weights derivatives for each segment
    eps::T
    min_dim::Int64
    min_pts::Int64
end

function CFIE(pts_scaling_factor::Union{T,Vector{T}},ws::Vector{Function},ws_der::Vector{Function},billiard::Bi;min_pts=20,fundamental::Bool=true,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    n_curves=fundamental ? length(billiard.fundamental_boundary) : length(billiard.full_boundary)
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor for _ in 1:n_curves] : pts_scaling_factor
    sampler=[LinearNodes() for _ in 1:n_curves] # placeholder for sampler, since we will rescale the quadrature weights
    return CFIE{T}(fundamental,sampler,bs,ws,ws_der,eps,min_pts,min_pts)
end

function CFIE(pts_scaling_factor::Union{T,Vector{T}},billiard::Bi;min_pts=20,q::Int=8,fundamental::Bool=true,eps=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    n_curves=fundamental ? length(billiard.fundamental_boundary) : length(billiard.full_boundary)
    bs=typeof(pts_scaling_factor)==T ? [pts_scaling_factor for _ in 1:n_curves] : pts_scaling_factor # one needs to be careful there are enough bs for all segments
    sampler=[LinearNodes() for _ in 1:n_curves]
    ws::Vector{Function}=[v->w_reparametrized(v,q) for _ in 1:n_curves] # quadrature weights for each segment, must be same length as the length of "fundamental::Bool" boundary, if true same as fundamental boundary, otherwise full boundary
    ws_der::Vector{Function}=[v->dw_reparametrized(v,q) for _ in 1:n_curves] # quadrature weights derivatives for each segment
    return CFIE{T}(fundamental,sampler,bs,ws,ws_der,eps,min_pts,min_pts)
end

#############################
#### BOUNDARY EVALUATION ####
#############################

struct BoundaryPointsCFIE{T}<:AbsPoints where {T<:Real}
    xy::Vector{SVector{2,T}} # the xy coords of the new mesh points
    normal::Vector{SVector{2,T}} # normals evaluated at the new mesh points
    curvature::Vector{T} # curvature evaluated at new mesh points
    sk::Vector{T} # new mesh points by w in solver
    sk_local::Vector{Vector{T}} # the local mesh points in [0,1] parametrizations for each segment
    ak::Vector{T} # the new weights (derivatives) of the new mesh points
end

function evaluate_points(solver::CFIE,billiard::Bi,k) where {Bi<:AbsBilliard}
    two_pi=2*pi
    fundamental=solver.fundamental
    boundary=fundamental ? billiard.fundamental_boundary : billiard.full_boundary
    Ls=[crv.length for crv in boundary];L=sum(Ls);Ls_scaled=Ls./L
    type=typeof(Ls_scaled[1])
    bs=solver.pts_scaling_factor
    Ns=[max(solver.min_pts,round(Int,k*Ls[i]*bs[i]/(two_pi))) for i in eachindex(Ls)];Ntot=sum(Ns)
    ts_all=midpoints(range(0,two_pi,length=(Ntot+1)))
    cuts=cumsum([0;Ns])
    ts_per_panel=[ts_all[cuts[i]+1:cuts[i+1]] for i in eachindex(Ns)]
    ws=solver.ws # we need only for adjacent segments the unique qaudrature 
    ws_der=solver.ws_der # derivativs of them
    xy_all=Vector{SVector{2,type}}()
    normal_all=Vector{SVector{2,type}}()
    kappa_all=Vector{type}()
    sk_all=Vector{type}()
    sk_local_all=Vector{Vector{type}}() # local mesh points in [0,1] parametrization for each segment
    ak_all=Vector{type}()
    for i in eachindex(ts_per_panel) 
        crv=boundary[i]
        t=ts_per_panel[i] 
        if i==1
            t_i=zero(type);t_f=two_pi*Ls_scaled[i] # the previous segments is the end segment and it's end parametrization is 0.0, so t_i=0.0
        else
            t_i=two_pi*Ls_scaled[i-1];t_f=two_pi*Ls_scaled[i] # the start and end of the segment in global parametrization
        end
        t_scaled=(t.-t_i)./(t_f-t_i) # need to rescale to ts_per_panel to local [0,1] parametrization since the ws and ws_der applied locally
        sk_local=ws[i](t_scaled) # we need to evaluate the sk first locally since the ws[i] is a local function (each segment has its own quadrature) and then project it to a global parameter
        sk=t_i.+sk_local.*(t_f-t_i) # now we can project it to the global parameter (w : [0,1] -> [0,1])
        ak=ws_der[i](t_scaled) # the weights of the new mesh points in the local coordinates
        xy=curve(crv,t_scaled) # the xy coordinates of the new mesh points, these are global now
        normal=normal_vec(crv,t_scaled) # the normals evaluated at the new mesh points, these are global now
        kappa=curvature(crv,t_scaled) # the curvature evaluated at the new mesh points, these are global now
        append!(xy_all,xy)
        append!(normal_all,normal)
        append!(kappa_all,kappa)
        append!(sk_all,sk)
        push!(sk_local_all,sk_local) # need to add as Vector, not splated with append!
        append!(ak_all,ak)
    end
    return BoundaryPointsCFIE(xy_all,normal_all,kappa_all,sk_all,sk_local_all,ak_all)
end

####################
#### CFIE UTILS ####
####################

function plot_boundary_with_weight_INFO(billiard::Bi,solver::CFIE;k=20.0) where {Bi<:AbsBilliard}
    pts=evaluate_points(solver,billiard,k)
    xs=getindex.(pts.xy,1)
    ys=getindex.(pts.xy,2)
    ak=pts.ak
    m=div(length(solver.ws),2)
    f=Figure(size=(1000+550*2,height=550*m),resolution=(1000+550*2,height=550*m))
    ax=Axis(f[1,1][1,1],title="Boundary with weights",width=1000,height=1000)
    scatter!(ax,xs,ys;markersize=4,color=ak,colormap=:viridis,strokewidth=0) #  colour by ak so you see where points are denser
    ws_ders=solver.ws_der
    r,c=1,1
    for (i,wder) in enumerate(solver.ws)
        if c>2
            r+=1;c=1
        end
        tloc=range(0.0,1.0,length=200)
        wline=wder(tloc)
        wderline=ws_ders[i](tloc)
        ax=Axis(f[1,2][r,c][1,1],width=500,height=500)
        lines!(ax,tloc,wline;label="panel $i",linewidth=2)
        axislegend(ax;position=:lt)
        ax=Axis(f[1,2][r,c][1,2],width=500,height=500)
        lines!(ax,tloc,wderline;label="panel $i derivative",linewidth=2)
        axislegend(ax;position=:lt)
        c+=1
    end
    return f
end