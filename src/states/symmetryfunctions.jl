
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

function symmetrize_wavefunction(Psi,x_grid,y_grid, symmetries::Vector{BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where T<:Real
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
