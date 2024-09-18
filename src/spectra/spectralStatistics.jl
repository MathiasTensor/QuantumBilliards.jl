using Polynomials

"""
INTERNAL FOR PLOTTING. Uses the def of σ² = (mean of squares) - (mean)^2 to determine the NV of a vector of unfolded energies in the energy window L
"""
function number_variance(E::Vector{T}, L::T) where {T<:Real}
    # By Tomaž Prosen
    Ave1 = zero(T)
    Ave2 = zero(T)
    j = 2
    k = 2
    x = E[1]
    N = length(E)
    largest_energy = E[end - min(Int(ceil(L) + 10), N-1)]  # Ensure largest_energy is within bounds

    while x < largest_energy
        while k < N && E[k] < x + L  # Ensure k does not exceed bounds with k<N = num of energies
            k += 1
        end

        # Adjusting the interval so that it slides/moves across the energy spectrum, a moving window statistic
        
        d1 = E[j] - x # The difference between the start of interval x and the first energy in the interval E[j]. This difference indicates how much the interval [x, E[j]] deviates from the exact interval [x, x + L]. d1 > 0
        d2 = E[k] - (x + L) # Difference between the end of the interval x + L and the first energy level beyond the interval [E[k]]. This difference shows how far the last energy level included in the interval is from the exact boundary x + L. d2 > 0
        cn = k - j # The number of energy levels in the interval (num of indexes for energies in the interval)
        
        # Interval Adjustment (d1 < d2):
        if d1 < d2
            # If the difference d1 (between x and E[j]) is smaller than d2 (between E[k] and x + L), the interval is adjusted by moving x to the next energy level E[j].
            x = E[j]
            # Since the difference between x and E[j] is smaller set the shift of the interval for x
            s = d1
            # Since d1 was smaller the updated x was for the start o the interval associated with the j index, so +1 it
            j += 1
        else
            # d2 was smaller so the new x is updated at the back of the interval
            x = E[k] - L
            # Analogues to the up case, just the smaller shift stored was d2
            s = d2
            # k is associated with the end interval
            k += 1
        end

        # Accumulations:
        # Ave1 = (1 / Total Length) * Σ(s_i * n(x_i, L)) where the number of energies in the interval n(x_i, L) is given by cn=k-j
        # Ave2 = (1 / Total Length) * Σ(s_i * n(x_i, L)) where the number of energies in the interval n(x_i, L) is given by cn=k-j
        Ave1 += s * cn
        Ave2 += s * cn^2

        # Ensure j does not exceed bounds
        if j >= N || k >= N # This condition checks if either j or k have reached or exceeded the total number of energy levels (N), j >= N || k >= N: If either index exceeds the bounds of the array, the loop is terminated to prevent out-of-bounds errors.
            break
        end
    end

    # See the formula at accumulations for reasons.
    total_length = largest_energy - E[1]
    Ave1 /= total_length
    Ave2 /= total_length

    # Calculate the variance σ² using the accumulated values Ave1 and Ave2. 
    # Variance = mean of squares - (mean)^2
    AveSig = Ave2 - Ave1^2
    return AveSig
end

"""
INTERNAL FOR PLOTTING. Uses a modified ver of Prosen's NV algorithm to determine the Δ3(L) for unfolded energies in a energy window of size L 
"""
function spectral_rigidity_new_parallel(E::Vector{T}, L::T) where {T<:Real}
    N = length(E)
    Ave = Threads.Atomic{T}(0.0)  # Use atomic operations to safely update the shared variable
    largest_energy = E[end - min(Int(ceil(L) + 10), N-1)]  # Ensure largest_energy is within bounds
    
    # Parallelize the main loop
    Threads.@threads for idx in 1:N
        j = idx
        k = j
        x = E[j]
        while x < largest_energy && j < N && k < N
            while k < N && E[k] < x + L  # Ensure k does not exceed bounds
                k += 1
            end
            
            d1 = E[j] - x
            d2 = E[k] - (x + L)
            cn = k - j  # Number of levels in the interval n(x_i, L)
            
            if cn < 2
                if d1 < d2
                    x = E[j]
                    j += 1
                else
                    x = E[k] - L
                    k += 1
                end
                continue
            end

            # Get the energy levels in the current interval
            E_interval = E[j:k-1]
            nE = 1:length(E_interval)

            # Perform a linear fit to n(E) = a + b*E
            p = Polynomials.fit(E_interval, nE, 1)
            
            # Calculate the deviation from the linear fit
            fit_values = [p(e) for e in E_interval]
            deviation = sum((nE .- fit_values).^2) / L

            # Accumulate the deviation for the interval
            s = min(d1, d2)
            Threads.atomic_add!(Ave, s * deviation)

            # Move to the next interval
            if d1 < d2
                x = E[j]
                j += 1
            else
                x = E[k] - L
                k += 1
            end
        end
    end

    # Normalize by the total length of the energy spectrum considered
    total_length = largest_energy - E[1]
    Ave_value = Ave[] / total_length

    return Ave_value  # This is the spectral rigidity, Δ3(L)
end
