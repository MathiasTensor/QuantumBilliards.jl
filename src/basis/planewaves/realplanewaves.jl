
struct RealPlaneWaves{T,Sa} <: AbsBasis where  {T<:Real, Sa<:AbsSampler}
    #cs::PolarCS{T} #not fully implemented
    dim::Int64 #using concrete type
    symmetries::Union{Vector{Any},Nothing}
    angle_arc::T
    angle_shift::T
    angles::Vector{T}
    parity_x::Vector{Int64}
    parity_y::Vector{Int64}
    sampler::Sa
end

function parity_pattern(symmetries)
    # Default parity vectors assuming no symmetries
    parity_x = [1, 1, -1, -1]
    parity_y = [1, -1, 1, -1]
    
    # Flags to track whether the axes are reflected
    x_reflected = false
    y_reflected = false
    xy_reflected = false
    
    # Loop over the symmetries
    if !isnothing(symmetries)
        for sym in symmetries
            if sym.axis == :y_axis
                # X-axis reflection: constrain parity_x
                parity_x = [sym.parity, sym.parity]
                x_reflected = true
            elseif sym.axis == :x_axis
                # Y-axis reflection: constrain parity_y
                parity_y = [sym.parity, sym.parity]
                y_reflected = true
            elseif sym.axis == :origin
                # XYReflection: constrain both axes to the same parity
                parity_x = [sym.parity]
                parity_y = [sym.parity]
                xy_reflected = true
                break  # Once XYReflection is applied, we must exit as there is nothing more to be done
            end
        end
    end

    if xy_reflected
        # XYReflection overrides any previous reflections, so lengths are already correct and we can terminate early
        return parity_x, parity_y
    end

    # Ensure both parity_x and parity_y have the same length to avoid problems of taking lenght of them for effective dimension
    if x_reflected && !y_reflected
        # If only XReflection is applied, adjust parity_y to match parity_x
        parity_y = [1, -1]
    elseif y_reflected && !x_reflected
        # If only YReflection is applied, adjust parity_x to match parity_y
        parity_x = [1, -1]
    end
    return parity_x, parity_y
end

function RealPlaneWaves(dim, symmetries; angle_arc = pi, angle_shift=0.0, sampler=LinearNodes())
    par_x, par_y = parity_pattern(symmetries)
    pl = length(par_x)
    eff_dim = dim*pl
    t, dt = sample_points(sampler, dim)
    angles = @. t*angle_arc + angle_shift
    angles = repeat(angles, inner=pl)
    par_x = repeat(par_x, outer=dim)
    par_y = repeat(par_y, outer=dim)
    return RealPlaneWaves(eff_dim, symmetries, angle_arc, angle_shift, angles, par_x, par_y, sampler)
end

function RealPlaneWaves(dim; angle_arc = pi, angle_shift=0.0, sampler=LinearNodes())
    symmetries = nothing
    par_x, par_y = parity_pattern(symmetries)
    pl = length(par_x)
    eff_dim = dim*pl
    t, dt = sample_points(sampler, dim)
    angles = @. t*angle_arc + angle_shift
    angles = repeat(angles, inner=pl)
    par_x = repeat(par_x, outer=dim)
    par_y = repeat(par_y, outer=dim)
    return RealPlaneWaves{eltype(angles),Nothing,typeof(sampler)}(eff_dim, symmetries, angle_arc, angle_shift, angles, par_x, par_y, sampler)
end

function resize_basis(basis::Ba, billiard::Bi, dim::Int, k) where {Ba<:RealPlaneWaves,Bi<:AbsBilliard}
    return RealPlaneWaves(dim, basis.symmetries; angle_arc = basis.angle_arc, angle_shift=basis.angle_shift, sampler=basis.sampler)
end

@inline function rpw(arg, parity::Int64)
    if parity == 1
        return cos.(arg)
    else
        return sin.(arg)
    end 
end

@inline function d_rpw(arg, parity::Int64)
    if parity == 1
        return -sin.(arg)
    else
        return cos.(arg)
    end 
end

@inline function basis_fun(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        vx = cos(basis.angles[i])
        vy = sin(basis.angles[i])
        arg_x = k*vx.*x
        arg_y = k*vy.*y
        b = rpw(arg_x, par_x[i]).*rpw(arg_y, par_y[i])
        return b
    end
end

@inline function basis_fun(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        M =  length(pts)
        N = length(indices)
        B = zeros(T,M,N)
        Threads.@threads for i in eachindex(indices)
            vx = cos(basis.angles[i])
            vy = sin(basis.angles[i])
            arg_x = k*vx.*x
            arg_y = k*vy.*y
            B[:,i] .= rpw(arg_x, par_x[i]).*rpw(arg_y, par_y[i])
        end
        return B 
    end
end

function gradient(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        vx = cos(basis.angles[i])
        vy = sin(basis.angles[i])
        arg_x = k*vx.*x
        arg_y = k*vy.*y
        bx = rpw(arg_x, par_x[i])
        by = rpw(arg_y, par_y[i])
        dx = k*vx.*d_rpw(arg_x, par_x[i]).*by
        dy = bx.*k*vy.*d_rpw(arg_y, par_y[i])
        return dx, dy
    end
end

function gradient(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        M =  length(pts)
        N = length(indices)
        dB_dx = zeros(T,M,N)
        dB_dy = zeros(T,M,N)
        Threads.@threads for i in eachindex(indices)
            vx = cos(basis.angles[i])
            vy = sin(basis.angles[i])
            arg_x = k*vx.*x
            arg_y = k*vy.*y
            bx = rpw(arg_x, par_x[i])
            by = rpw(arg_y, par_y[i])
            dB_dx[:,i] .= k*vx.*d_rpw(arg_x, par_x[i]).*by
            dB_dy[:,i] .= bx.*k*vy.*d_rpw(arg_y, par_y[i])
        end
        return dB_dx, dB_dy
    end
end


function basis_and_gradient(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        vx = cos(basis.angles[i])
        vy = sin(basis.angles[i])
        arg_x = k*vx.*x
        arg_y = k*vy.*y
        bx = rpw(arg_x, par_x[i])
        by = rpw(arg_y, par_y[i])
        bf = bx.*by
        dx = k*vx.*d_rpw(arg_x, par_x[i]).*by
        dy = bx.*k*vy.*d_rpw(arg_y, par_y[i])
        return bf, dx, dy
    end
end


function basis_and_gradient(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        M =  length(pts)
        N = length(indices)
        B = zeros(T,M,N)
        dB_dx = zeros(T,M,N)
        dB_dy = zeros(T,M,N)
        Threads.@threads for i in eachindex(indices)
            vx = cos(basis.angles[i])
            vy = sin(basis.angles[i])
            arg_x = k*vx.*x
            arg_y = k*vy.*y
            bx = rpw(arg_x, par_x[i])
            by = rpw(arg_y, par_y[i])
            B[:,i] .= bx.*by
            dB_dx[:,i] .= k*vx.*d_rpw(arg_x, par_x[i]).*by
            dB_dy[:,i] .= bx.*k*vy.*d_rpw(arg_y, par_y[i])

        end
        return B, dB_dx, dB_dy
    end
end

@inline function dk_fun(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        vx = cos(basis.angles[i])
        vy = sin(basis.angles[i])
        arg_x = k*vx.*x
        arg_y = k*vy.*y
        bx = rpw(arg_x, par_x[i])
        by = rpw(arg_y, par_y[i])
        d_bx = d_rpw(arg_x, par_x[i])
        d_by = d_rpw(arg_y, par_y[i])
        dk = @. vx*x*d_bx*by + bx*vy*y*d_by
        return dk
    end
end
    

@inline function dk_fun(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray) where {T<:Real}
    let par_x = basis.parity_x, par_y = basis.parity_y
        x = getindex.(pts,1)
        y = getindex.(pts,2)
        M =  length(pts)
        N = length(indices)
        dB_dk = zeros(T,M,N)
        Threads.@threads for i in eachindex(indices)
            vx = cos(basis.angles[i])
            vy = sin(basis.angles[i])
            arg_x = k*vx.*x
            arg_y = k*vy.*y
            bx = rpw(arg_x, par_x[i])
            by = rpw(arg_y, par_y[i])
            d_bx = d_rpw(arg_x, par_x[i])
            d_by = d_rpw(arg_y, par_y[i])
            dB_dk[:,i] .=  @. vx*x*d_bx*by + bx*vy*y*d_by
        end
        return dB_dk
    end
end
