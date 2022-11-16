@doc raw"""
    truncate!(P, PE, Y::TruncationMetric; gauge=2)

Truncate the tensor `PE` along the east-west bond according the truncation metric `Y`, and 
store the result in `P`.
"""
function truncate!(P,PE,Y::TruncationMetric; ξ=Inf, gauge=2)
    truncspace = codomain(P[1,1])[1] # The space we wish to truncate down to.

    # Optimise the isometries to maximise fidelity.
    U,S,V, = optimise_isometries(Y, truncspace)

    # Convert the isometries to the weighted trace gauge 
    if !(gauge == 0)
        U, S, V = wtg!(U, S, V, Y; ξ=ξ,test=false)
        
        if !(gauge == 1) 
            left, right = wtg_test(U, S, V, Y)
            if !left
                @warn "Left weighted trace gauge condition was not met"
            end
            if !right
                @warn "Right weighted trace gauge condition was not met"
            end
        end
    end

    # Truncate the bonds using the isometries
    isometry_update!(P, PE, U, S, V)
    
    return P
end
