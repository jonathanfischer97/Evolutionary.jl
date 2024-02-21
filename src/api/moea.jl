"""
    dominate(p, q)

Determines the dominance relationship between two solutions `p` and `q`.

# Arguments
- `p::T`: A solution represented as an abstract array.
- `q::T`: Another solution represented as an abstract array.

# Returns
- `1` if `p` is dominated by `q` (i.e., `q` is better than `p` in all objectives).
- `-1` if `p` dominates `q` (i.e., `p` is better than `q` in all objectives).
- `0` if neither solution dominates the other (i.e., the solutions are non-dominating).

This function iterates over each objective of the solutions to determine the dominance relationship.
"""
function dominate(p::T, q::T) where {T <: AbstractArray}
    ret = 0  # Default return value indicating no dominance
    for (i,j) in zip(p,q)  # Iterate over each objective of the solutions
        if i < j  # If objective of `p` is worse than `q`
            ret == -1 && return 0  # If `p` was previously found to dominate `q`, return 0 (non-dominating)
            ret = 1  # Mark `p` as being dominated by `q`
        elseif j < i  # If objective of `q` is worse than `p`
            ret == 1 && return 0  # If `p` was previously found to be dominated by `q`, return 0 (non-dominating)
            ret = -1  # Mark `p` as dominating `q`
        end
    end
    return ret  # Return the dominance relationship
end

"""
    dominations(P::AbstractVector)

Calculates and returns a domination matrix for a given population `P`.

The domination matrix `D` is a square matrix where each element `D[i,j]` represents the dominance relationship between individuals `i` and `j` in the population. The value of `D[i,j]` is:
- `1` if individual `i` is dominated by individual `j`,
- `-1` if individual `i` dominates individual `j`,
- `0` if neither individual dominates the other.

# Arguments
- `P::AbstractVector`: A vector of individuals in the population, where each individual is represented as an abstract array.

# Returns
- `D::Matrix{Int8}`: A square matrix of dominance relationships among all individuals in `P`.
"""
function dominations(P::AbstractVector{T}) where {T <: AbstractArray}
    l = length(P)  # Get the number of individuals in the population
    D = zeros(Int8, l, l)  # Initialize the domination matrix with zeros
    for i in 1:l  # Iterate over each individual in the population
        for j in (i+1):l  # Compare with every other individual
            D[i,j] = dominate(P[i],P[j])  # Determine if `i` is dominated by `j`
            D[j,i] = -D[i,j]  # The opposite relationship is the negative value
        end
    end
    D  # Return the domination matrix
end

"""
    nondominatedsort!(R, P)

Calculate fronts for fitness values `F`, and store ranks of the individuals into `R`.

# Arguments
- `R`: A vector to store the rank of each individual.
- `P`: A matrix where each column represents an individual and rows represent objectives.

# Notes
This function modifies `R` in place to reflect the ranks based on non-dominated sorting.
"""
function nondominatedsort!(R, P)
    n = size(P,2)  # Number of individuals
    @assert length(R) == n "Ranks must be defined for the whole population"

    # Initialize a dictionary to keep track of sets of individuals that an individual dominates
    Sₚ = Dict(i=>Set() for i in 1:n)
    # Initialize a count of how many individuals dominate each individual
    C = zeros(Int, n)

    # Initialize the first front as an empty list
    F =[Int[]]
    for i in 1:n  # For each individual
        for j in i+1:n  # Compare with every other individual
            r = dominate(view(P,:,i), view(P,:,j))  # Determine dominance relationship
            if r == 1  # If i is dominated by j
                push!(Sₚ[i], j)  # Add j to the set of individuals that i dominates
                C[j] += 1  # Increment domination count for j
            elseif r == -1  # If i dominates j
                push!(Sₚ[j], i)  # Add i to the set of individuals that j dominates
                C[i] += 1  # Increment domination count for i
            end
        end
        if C[i] == 0  # If individual i is not dominated by any other
            R[i] = 1  # Assign rank 1 (first front)
            push!(F[1], i)  # Add i to the first front
        end
    end

    # Construct the rest of the fronts
    while !isempty(last(F))  # While the last front is not empty
        Q = Int[]  # Initialize next front
        for i in last(F)  # For each individual in the last front
            for j in Sₚ[i]  # For each individual that i dominates
                C[j] -= 1  # Decrement domination count for j
                if C[j] == 0  # If j is not dominated by any other individuals
                    push!(Q, j)  # Add j to the next front
                    R[j] = length(F) + 1  # Assign rank based on the next front's position
                end
            end
        end
        push!(F, Q)  # Add the next front to the list of fronts
    end
    isempty(last(F)) && pop!(F)  # Remove the last front if it is empty

    F  # Return the list of fronts
end

"""
    crowding_distance!(C, F, fronts)

Calculate crowding distance for individuals and save the results into `C`
given the fitness values `F` and collection of `fronts`.
"""
function crowding_distance!(C::AbstractVector, F::AbstractMatrix{T}, fronts) where {T}
    # Iterate through each front
    for f in fronts
        # Get a view of the crowding distances for the current front
        cf = @view C[f]
        # If the front has 2 or fewer individuals, assign them the maximum possible distance
        if length(cf) <= 2
            cf .= typemax(T)
        else
            # Extract the fitness values for the current front
            SF = F[:, f]
            # Get the number of objectives
            d = size(SF,1)
            # Initialize arrays for sorting and reordering
            IX = zeros(Int, size(SF))
            IIX = zeros(Int, size(SF))
            # Sort each objective and calculate distances
            for i in 1:d
                # Get views for sorting and reordering
                irow, iirow, row = view(IX,i,:), view(IIX,i,:), view(SF,i,:)
                # Sort the individuals based on the current objective
                sortperm!(irow, row)
                # Get the inverse permutation order
                sortperm!(iirow, irow)
                # Reorder the fitness values based on the sorted indices
                permute!(row, irow)
            end
            # Calculate the normalization factor as the difference between the max and min values
            nrm = SF[:,end] - SF[:,1]
            # Calculate the distance for each individual based on their normalized fitness values
            dst = (hcat(SF, fill(typemax(T), d)) - hcat(fill(typemin(T), d), SF)) ./ nrm
            # Replace NaN values with zero
            dst[isnan.(dst)] .= zero(T)
            # Sum the distances for each individual and normalize by the number of objectives
            ss = sum(mapslices(v->diag(dst[:,v]) + diag(dst[:,v.+1]), IIX, dims=1), dims=1)
            # Assign the calculated crowding distances to the individuals in the current front
            cf .= vec(ss)/d
        end
    end
    # Return the updated crowding distances
    C
end

