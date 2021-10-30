module Evolutionary
    using Random, LinearAlgebra, Statistics
    using Base: @kwdef
    using UnPack: @unpack
    using NLSolversBase: NLSolversBase, AbstractObjective, ConstraintBounds,
                         AbstractConstraints, nconstraints_x, nconstraints

    import NLSolversBase: NonDifferentiable, f_calls, value, value!
    import Base: show, copy, minimum, summary, identity, getproperty, rand, getindex, length, copyto!, setindex!

    export AbstractStrategy, strategy, mutationwrapper,
           IsotropicStrategy, AnisotropicStrategy, NoStrategy,
           isfeasible, BoxConstraints, apply!, penalty, penalty!,
           PenaltyConstraints, WorstFitnessConstraints, MixedTypePenaltyConstraints,
           # ES mutations
           gaussian, cauchy,
           # GA mutations
           flip, bitinversion, uniform, domainrange, inversion, insertion, swap2, scramble, shifting, PM, MIPM,
           # ES recombinations
           average, marriage,
           # GA recombinations
           SPX, TPX, UX,
           discrete, waverage, intermediate, line, HX, LX, MILX, SBX,
           PMX, OX1, CX, OX2, POS,
           # GA selections
           ranklinear, uniformranking, roulette, rouletteinv, sus, susinv, tournament, truncation,
           # DE selections
           random, permutation, randomoffset, best,
           # DE recombinations
           BINX, EXPX,
           # GP exports
           Terminal, subtree, point, hoist, shrink,
           # Optimization methods
           ES, CMAES, GA, DE, TreeGP,
           # re-export
           NonDifferentiable, value, value!

    # optimize API
    include("api/types.jl")
    include("api/results.jl")
    include("api/utilities.jl")
    include("api/constraints.jl")
    include("api/optimize.jl")
    include("api/expressions.jl")

    # Evolution Strategy
    include("es.jl")
    include("cmaes.jl")

    # Genetic Algorithms
    include("ga.jl")

    # Differential Evolution
    include("de.jl")

    # Genetic Programming
    include("api/protected.jl")
    include("gp.jl")

    # ES & GA recombination functions
    include("recombinations.jl")

    # ES & GA mutation functions
    include("mutations.jl")

    # GA selection functions
    include("selections.jl")

    @deprecate uniform(v1, v2) UX(v1, v2)
    @deprecate uniformbin BINX
    @deprecate exponential EXPX
    @deprecate singlepoint SPX
    @deprecate twopoint TPX

end
