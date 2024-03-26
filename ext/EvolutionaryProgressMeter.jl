module EvolutionaryProgressMeter

    # using Evolutionary, ProgressMeter
    import Evolutionary
    using ProgressMeter

    function Evolutionary.value!(obj::Evolutionary.EvolutionaryObjective{TC,TF,TX,Val{:threadprogress}},
        F::AbstractVector, xs::AbstractVector{TX}) where {TC,TF<:Real,TX}

        n = length(xs)
        @showprogress Threads.@threads for i in 1:n
            F[i] = Evolutionary.value(obj, xs[i])
            end
        F
    end

end