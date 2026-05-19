
function apply_symmetries_to_wavefunction(
    Psi, x_grid, y_grid,
    symmetries::Vector{<:BilliardGeometry.AbsReflection},
    sym_qnumbers::Vector{T}
) where T<:Real

    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)

    get_qnum(::Type{S}) where S = sym_qnumbers[findfirst(s -> s isa S, symmetries)]

    if has_x && has_y
        pX = get_qnum(BilliardGeometry.XAxisReflection)
        pY = get_qnum(BilliardGeometry.YAxisReflection)

        # Q1(base) → Q2(Y-reflect) → Q3(XY-reflect) → Q4(X-reflect)
        # expand along x first (Y-axis reflection: reflects x coordinates)
        x_ref  = -reverse(x_grid)
        Psi_xref = reverse(pY .* Psi; dims=1)
        Psi_x  = hcat(Psi_xref, Psi) # wait - need to think in terms of matrix dims

        # Psi is (nx, ny): rows=x, cols=y
        # YAxisReflection: mirror x → prepend reversed rows, pY
        Psi_Y   = reverse(pY .* Psi;  dims=1)   # Q2
        Psi_XY  = reverse(pX .* Psi_Y; dims=2)  # Q3: also mirror y
        Psi_X   = reverse(pX .* Psi;  dims=2)   # Q4

        # CCW order: Q2(x-reversed), Q1(base) along x axis
        # then Q3, Q4 along y axis
        full_Psi = vcat(Psi_Y, Psi)               # expand x: [Q2; Q1]
        full_Psi_XY = vcat(Psi_XY, Psi_X)         # expand x: [Q3; Q4]
        full_Psi = hcat(full_Psi_XY, full_Psi)    # expand y: [Q3,Q4 | Q2,Q1] wait...

        x_grid = vcat(-reverse(x_grid), x_grid)
        y_grid = vcat(-reverse(y_grid), y_grid)

        return full_Psi, x_grid, y_grid
    elseif has_y
        p = get_qnum(BilliardGeometry.YAxisReflection)
        Psi_ref = reverse(p .* Psi; dims=2)
        return hcat(Psi_ref, Psi), x_grid, vcat(-reverse(y_grid), y_grid)
    elseif has_x
        p = get_qnum(BilliardGeometry.XAxisReflection)
        Psi_ref = reverse(p .* Psi; dims=1)
        return vcat(Psi_ref, Psi), vcat(-reverse(x_grid), x_grid), y_grid
    else
        return Psi, x_grid, y_grid
    end
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

    @inline function push_reflection!(sym::BilliardGeometry.AbsReflection, reverse_orientation::Bool)
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