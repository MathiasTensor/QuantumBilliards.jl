
function apply_symmetry_wf(Psi, x_grid, y_grid, sym::BilliardGeometry.XAxisReflection, qnumber::T) where T<:Real
    # compute tolerance from grid scale for zero-comparison
    scale = maximum(abs.(y_grid))
    tol = 1e-12 * max(scale, 1.0)

    # If the original grid contains the axis point, remove that column/point
    idx0 = findfirst(y -> isapprox(y, 0; atol=tol), y_grid)
    if idx0 !== nothing
        # remove axis column from original Psi and the axis point from y_grid
        if idx0 == 1
            Psi = Psi[:, 2:end]
            y_grid = y_grid[2:end]
        elseif idx0 == size(Psi, 2)
            Psi = Psi[:, 1:end-1]
            y_grid = y_grid[1:end-1]
        else
            Psi = hcat(Psi[:, 1:idx0-1], Psi[:, idx0+1:end])
            y_grid = vcat(y_grid[1:idx0-1], y_grid[idx0+1:end])
        end
    end

    # reflect remaining grid and Psi across y (reverse columns)
    y_ref = -reverse(y_grid)
    Psi_ref = reverse(qnumber .* Psi; dims=2)

    Psi = hcat(Psi_ref, Psi)
    y_grid = append!(y_ref, y_grid)
    return Psi, x_grid, y_grid
end

function apply_symmetry_wf(Psi, x_grid, y_grid, sym::BilliardGeometry.YAxisReflection, qnumber::T) where T<:Real
    # compute tolerance from grid scale for zero-comparison
    scale_x = maximum(abs.(x_grid))
    tol_x = 1e-12 * max(scale_x, 1.0)

    # If the original grid contains the axis point, remove that row/point
    idx0x = findfirst(x -> isapprox(x, 0; atol=tol_x), x_grid)
    if idx0x !== nothing
        # remove axis row from original Psi and the axis point from x_grid
        if idx0x == 1
            Psi = Psi[2:end, :]
            x_grid = x_grid[2:end]
        elseif idx0x == size(Psi, 1)
            Psi = Psi[1:end-1, :]
            x_grid = x_grid[1:end-1]
        else
            Psi = vcat(Psi[1:idx0x-1, :], Psi[idx0x+1:end, :])
            x_grid = vcat(x_grid[1:idx0x-1], x_grid[idx0x+1:end])
        end
    end

    # reflect remaining grid and Psi across x (reverse rows)
    x_ref = -reverse(x_grid)
    Psi_ref = reverse(qnumber .* Psi; dims=1)

    Psi = vcat(Psi_ref, Psi)
    x_grid = append!(x_ref, x_grid)
    return Psi, x_grid, y_grid
end

function apply_symmetries_to_wavefunction(Psi,x_grid,y_grid, symmetries::Vector{BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where T<:Real
    for (sym, qnumber) in zip(symmetries, sym_qnumbers)
        if sym  isa BilliardGeometry.XAxisReflection
            Psi, x_grid, y_grid = apply_symmetry_wf(Psi,x_grid,y_grid,sym,qnumber)
        end
        if sym  isa BilliardGeometry.YAxisReflection
            Psi, x_grid, y_grid = apply_symmetry_wf(Psi,x_grid,y_grid,sym,qnumber)
        end
    end
    return Psi, x_grid, y_grid
end

function apply_symmetries_to_boundary_function(
    u::AbstractVector{U},
    symmetries::Vector{<:BilliardGeometry.AbsReflection},
    sym_qnumbers::Vector{T}
) where {U<:Number, T<:Real}

    isempty(symmetries) && return u

    base_u = copy(u)
    full_u = copy(u)

    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)

    if has_x && has_y
        pY = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)]
        pX = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)]

        # CCW order must match apply_symmetries_to_boundary_points exactly:
        # Q1(base) → Q2(Y-reflect, reversed) → Q3(XY-reflect) → Q4(X-reflect, reversed)
        uY  =  pY      .* reverse(base_u)  # Q2: reversed
        uXY = (pX * pY) .*        base_u   # Q3: not reversed
        uX  =  pX      .* reverse(base_u)  # Q4: reversed

        append!(full_u, uY)
        append!(full_u, uXY)
        append!(full_u, uX)

    elseif has_y
        pY = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)]
        append!(full_u, pY .* reverse(base_u))

    elseif has_x
        pX = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)]
        append!(full_u, pX .* reverse(base_u))
    end

    return full_u
end


function apply_symmetries_to_boundary_points(
    pts::BoundaryPoints{T},
    symmetries::Vector{BilliardGeometry.AbsReflection},
    billiard::Bi
) where {Bi<:AbsBilliard, T<:Real}

    isempty(symmetries) && return pts

    bxy    = pts.xy
    bn     = pts.normal
    bds    = pts.ds

    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)

    copies = 1 + has_x + has_y + (has_x & has_y)

    full_xy     = copy(bxy)
    full_normal = copy(bn)
    full_ds     = copy(bds)
    sizehint!(full_xy,     length(bxy) * copies)
    sizehint!(full_normal, length(bn)  * copies)
    sizehint!(full_ds,     length(bds) * copies)

    get_sym(::Type{S}) where S = symmetries[findfirst(s -> s isa S, symmetries)]

    @inline function push_reflection!(sym::AbsReflection, reverse_orientation::Bool)
        rxy = apply_symmetry(sym, bxy)
        rn  = apply_symmetry(sym, bn)
        if reverse_orientation
            append!(full_xy,     reverse(rxy))
            append!(full_normal, reverse(rn))
            append!(full_ds,     reverse(bds))
        else
            append!(full_xy,     rxy)
            append!(full_normal, rn)
            append!(full_ds,     bds)
        end
        return nothing
    end

    if has_x && has_y
        # CCW order: Q1(base) → Q2(Y-reflect) → Q3(XY-reflect) → Q4(X-reflect)
        push_reflection!(get_sym(BilliardGeometry.YAxisReflection),  true)  # Q1→Q2: reverse
        push_reflection!(get_sym(BilliardGeometry.XYAxisReflection), false) # Q2→Q3: preserve
        push_reflection!(get_sym(BilliardGeometry.XAxisReflection),  true)  # Q3→Q4: reverse
    elseif has_y
        # CCW order: Q1(base) → Q2(Y-reflect, reversed)
        push_reflection!(get_sym(BilliardGeometry.YAxisReflection), true)
    elseif has_x
        # CCW order: Q1(base) → Q4(X-reflect, reversed)
        push_reflection!(get_sym(BilliardGeometry.XAxisReflection), true)
    end

    full_s = cumsum(full_ds)

    return BoundaryPoints{T}(
        full_xy, full_normal,
        T[], full_s, full_ds,
        T[], T[], T[],
        SVector{2,T}[]
    )
end