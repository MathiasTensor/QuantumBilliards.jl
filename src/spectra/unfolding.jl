using QuadGK

corner_correction(corner_angles) =  sum([(pi^2 - c^2)/(24*pi*c) for c in corner_angles])

function curvature_correction(billiard::Bi) where {Bi<:AbsBilliard}
    let segments = billiard.full_boundary
        println(segments)
        println("Number of segments: ", length(segments))
        for seg in segments 
            println(typeof(seg))
        end
        curvat = 0.0
        for seg in segments 
            if seg isa PolarSegment
                println("We are doing polar segment")
                curvat += quadgk(t -> curvature(seg, t), 0.0, 1.0)[1]
            end
            if seg isa CircleSegment
                println("We are doing circle segment")
                curvat += 1/(12*pi)*(1/seg.radius)*seg.length
            end
        end
        return curvat
    end
end

weyl_law(k,A,L) =  @. (A * k^2 - L * k)/(4*pi)
weyl_law(k,A,L,corner_angles) =  weyl_law(k,A,L) .+ corner_correction(corner_angles)


function k_at_state(state, A, L)
    a = A
    b = -L
    c = -state*4*pi
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end

function k_at_state(state, A, L, corner_angles)
    a = A
    b = -L
    c = (corner_correction(corner_angles)-state)*4*pi 
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end