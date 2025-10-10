"""
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T},
                                        symmetries::Union{Vector{Any},Nothing},
                                        billiard::Bi; same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}

Extend a desymmetrized set of boundary points by applying reflections/rotations.

- Uses `_x_reflect/_y_reflect` and `_x_reflect_normal/_y_reflect_normal/_xy_reflect_normal`.
- Reflections are around `x=billiard.x_axis` / `y=billiard.y_axis` if present (0 otherwise).
- If `same_direction=true`, reflected copies are reversed (xy, normal, and ds) to maintain CCW orientation.
- Rotations are about `sym.center`; `ds` is copied; `s` is rebuilt via `cumsum(ds)` at the end.

Returns a new `BoundaryPoints{T}` on the full, symmetry-extended boundary.
"""
function apply_symmetries_to_boundary_points(pts::BoundaryPoints{T},symmetries::Union{Vector{Any},Nothing},billiard::Bi;same_direction::Bool=true) where {Bi<:AbsBilliard,T<:Real}
    symmetries===nothing && return pts
    bxy=pts.xy
    bn=pts.normal
    bds=pts.ds
    full_xy=copy(bxy)
    full_normal=copy(bn)
    full_ds=copy(bds)
    # rough size hint
    copies=1+sum(s-> s isa Reflection ? (s.axis===:origin ? 3 : 1) : s isa Rotation ? (s.n-1) : 0,symmetries)
    sizehint!(full_xy,length(bxy)*copies)
    sizehint!(full_normal,length(bn)*copies)
    sizehint!(full_ds,length(bds)*copies)
    sx=hasproperty(billiard,:x_axis) ? T(billiard.x_axis) : zero(T)
    sy=hasproperty(billiard,:y_axis) ? T(billiard.y_axis) : zero(T)

    @inline function push_reflection!(which::Symbol)
        if which===:x
            rxy=[SVector(_x_reflect(p[1],sx),p[2]) for p in bxy]
            rn=[_x_reflect_normal(nv[1],nv[2]) for nv in bn]
        elseif which === :y
            rxy=[SVector(p[1], _y_reflect(p[2],sy)) for p in bxy]
            rn=[_y_reflect_normal(nv[1], nv[2]) for nv in bn]
        elseif which === :xy
            rxy=[SVector(_x_reflect(p[1],sx),_y_reflect(p[2], sy)) for p in bxy]
            rn=[_xy_reflect_normal(nv[1],nv[2]) for nv in bn]
        else
            error("unknown reflection kind $which")
        end
        # reverse only for single reflections; NOT for :xy
        do_reverse=same_direction && (which!=:xy)
        rds=do_reverse ? reverse(bds) : bds
        rxy=do_reverse ? reverse(rxy) : rxy
        rn=do_reverse ? reverse(rn)  : rn
        append!(full_xy,rxy)
        append!(full_normal,rn)
        append!(full_ds,rds)
        return nothing
    end
    for s in symmetries
        if s isa Reflection
            if s.axis===:y_axis
                push_reflection!(:x)
            elseif s.axis===:x_axis
                push_reflection!(:y)
            elseif s.axis===:origin
                push_reflection!(:x)
                push_reflection!(:y)
                push_reflection!(:xy)
            else
                error("Unknown reflection axis $(s.axis)")
            end
        elseif s isa Rotation
            n=s.n;cx,cy=s.center
            Cx=T(cx);Cy=T(cy);θ=T(2π)/T(n)
            for l in 1:n-1
                cl=cos(T(l)*θ);sl=sin(T(l)*θ)
                rxy=[SVector(cl*(p[1]-Cx)-sl*(p[2]-Cy)+Cx,sl*(p[1]-Cx)+cl*(p[2]-Cy)+Cy) for p in bxy]
                rn =[SVector(cl*nv[1]-sl*nv[2],sl*nv[1]+cl*nv[2]) for nv in bn]
                append!(full_xy,rxy)
                append!(full_normal,rn)
                append!(full_ds,bds)
            end
        else
            error("Unknown symmetry type: $(typeof(s))")
        end
    end
    full_s=cumsum(full_ds)
    return BoundaryPoints{T}(full_xy,full_normal,full_s,full_ds)
end

"""
    apply_symmetries_to_boundary_function(u::AbstractVector{U},
                                          symmetries::Union{Vector{Any},Nothing}) where {U<:Number}

Symmetrize the desymmetrized boundary function `u(s)` for the full boundary.
Works with real or complex `u`. If any rotation has `m % n ≠ 0`, a complex
character phase χ_l is applied and the result is `Vector{Complex{T}}`
(with `T = real(eltype(u))`); otherwise the element type is preserved.

Reflection rules:
- `:y_axis` or `:x_axis`: append `parity * reverse(u)`.
- `:origin`: append `parity[1] * reverse(u)` (vertical), then
  `parity[2] * reverse([u; that_vertical_reflection])` (horizontal of the combined).

Rotation rules:
- For `l=1..n-1`, append `χ_l * u`, where `χ_l = exp(i 2π m l / n)`.
  If `m % n == 0`, χ_l=1 and no complex promotion occurs.

Returns the concatenated full-boundary function.
"""
function apply_symmetries_to_boundary_function(u::AbstractVector{U},symmetries::Union{Vector{Any},Nothing}) where {U<:Number}
    symmetries===nothing && return u
    T=U<:Real ? U : eltype(real(zero(U)))
    has_complex=any(s->(s isa Rotation)&&mod(s.m,s.n)!=0,symmetries)
    S=(U<:Real && has_complex) ? Complex{T} : U
    full_u=S.(u)
    base_u=copy(full_u) # not alias for rotations
    for sym in symmetries
        if sym isa Reflection
            if sym.axis==:y_axis
            p=S(sym.parity)
            append!(full_u,p.*reverse(base_u))     # matches :x block in points
        elseif sym.axis==:x_axis
            p=S(sym.parity)
            append!(full_u,p.*reverse(base_u))     # matches :y block in points
        elseif sym.axis==:origin
            pY=S(sym.parity[1])  # parity for y-axis reflection (vertical)
            pX=S(sym.parity[2])  # parity for x-axis reflection (horizontal)
            uY=pY.*reverse(base_u)               #  :x block in points (y-axis reflection)
            uX=pX.*reverse(base_u)               #  :y block in points (x-axis reflection)
            uXY=(pX*pY).*base_u                   #  :xy block (no reverse!)
            append!(full_u,uY)
            append!(full_u,uX)
            append!(full_u,uXY)
        else
            error("Unknown reflection axis $(sym.axis)")
        end
        elseif sym isa Rotation
            n=sym.n; m=mod(sym.m,n)
            if m==0
                for l in 1:(n-1); append!(full_u,base_u); end # χ(m=0) = 1
            else
                for l in 1:(n-1)
                    θ=T(2π)*T(m*l)/T(n)
                    χ=Complex{T}(cos(θ),sin(θ))
                    append!(full_u,χ.*base_u)
                end
            end
        else
            error("Unknown symmetry type: $(typeof(sym))")
        end
    end
    return full_u
end

#### TESTING FUNCTION ####

"""
    difference_boundary_points(pts_new::BoundaryPoints{T}, pts_old::BoundaryPoints{T}) where {T<:Real}

Compute the difference between two `BoundaryPoints` objects. The returned `BoundaryPoints` object contains
only the elements in `pts_new` that are not present in `pts_old`. Throws errors if the lengths of the `Vector` fields in either the old or new inputs is not the same

# Arguments
- `pts_new::BoundaryPoints`: The updated `BoundaryPoints` object that contains all the points.
- `pts_old::BoundaryPoints`: The original `BoundaryPoints` object whose data is a subset of `pts_new`.

# Returns
- A new `BoundaryPoints` object containing the fields (`xy`, `normal`, `s`, `ds`) from `pts_new`
  starting from the index where `pts_old` ends.
"""
function difference_boundary_points(pts_new::BoundaryPoints{T},pts_old::BoundaryPoints{T}) where {T<:Real}
    if !(length(pts_new.xy)==length(pts_new.normal)==length(pts_new.s)==length(pts_new.ds))
        println("length xy: ",length(pts_new.xy))
        println("length normal: ",length(pts_new.normal))
        println("length s: ",length(pts_new.s))
        println("length ds: ",length(pts_new.ds))
        error("Fields of pts_new are not consistent in length")
    end
    if !(length(pts_old.xy)==length(pts_old.normal)==length(pts_old.s)==length(pts_old.ds))
        println("length xy: ",length(pts_old.xy))
        println("length normal: ",length(pts_old.normal))
        println("length s: ",length(pts_old.s))
        println("length ds: ",length(pts_old.ds))
        error("Fields of pts_old are not consistent in length")
    end
    start_index=length(pts_old.xy)+1
    new_xy=pts_new.xy[start_index:end]
    new_normal=pts_new.normal[start_index:end]
    new_s=pts_new.s[start_index:end]
    new_ds=pts_new.ds[start_index:end]
    return BoundaryPoints(new_xy,new_normal,new_s,new_ds)
end