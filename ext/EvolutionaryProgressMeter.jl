module EvolutionaryProgressMeter

    # using Evolutionary, ProgressMeter
    using ProgressMeter
    using Evolutionary
    import Evolutionary: value!

    function Evolutionary.value!(obj::Evolutionary.EvolutionaryObjective{TC,TF,TX,Val{:threadprogress}},
        F::AbstractVector, xs::AbstractVector{TX}) where {TC,TF<:Real,TX}

        n = length(xs)
        @showprogress Threads.@threads for i in 1:n
            F[i] = value(obj, xs[i])
            end
        F
    end

    function Evolutionary.value!(obj::Evolutionary.EvolutionaryObjective{TC,TF,TX,Val{:threadprogress}},
                F::AbstractMatrix, xs::AbstractVector{TX}) where {TC,TF,TX}

        n = length(xs)
        @showprogress Threads.@threads for i in 1:n
            fv = view(F, :, i)
            value(obj, fv, xs[i])
        end
        F
    end
end