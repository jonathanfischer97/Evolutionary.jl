module EvolutionaryProgressMeter

    # using Evolutionary, ProgressMeter
    using ProgressMeter: @showprogress
    using Evolutionary
    import Evolutionary: value!

    function Evolutionary.value!(obj::Evolutionary.EvolutionaryObjective{TC,TF,TX,Val{:thread}},
        F::AbstractVector, xs::AbstractVector{TX}) where {TC,TF<:Real,TX<:AbstractVector}

        n = length(xs)
        @showprogress Threads.@threads for i in 1:n
            F[i] = value(obj, xs[i])
            end
        F
    end

end