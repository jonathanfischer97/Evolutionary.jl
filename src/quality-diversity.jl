"""
Implementation of Quality Diversity algorithm (uses GA)

The constructor takes following keyword arguments:

- `populationSize`: The size of the population
- `crossoverRate`: The fraction of the population at the next generation, not including elite children, that is created by the crossover function.
- `mutationRate`: Probability of chromosome to be mutated
- `ɛ`/`epsilon`: Positive integer specifies how many individuals in the current generation are guaranteed to survive to the next generation.
Floating number specifies fraction of population.
- `selection`: [Selection](@ref) function (default: [`tournament`](@ref))
- `crossover`: [Crossover](@ref) function (default: [`genop`](@ref))
- `mutation`: [Mutation](@ref) function (default: [`genop`](@ref))
- `metrics` is a collection of convergence metrics.
- `optsys` is the OptimizationReactionSystem, or whatever struct is used to pass the ODE problem or any other miscellaneous data to the trace function.
"""
struct QD{T1,T2,T3,T4, T5} <: AbstractOptimizer
    populationSize::Int
    crossoverRate::Float64
    mutationRate::Float64
    ɛ::Real
    selection::T1
    crossover::T2
    mutation::T3
    metrics::T4
    optsys::T5

    QD(; populationSize::Int=10000, crossoverRate::Float64=0.75, mutationRate::Float64=0.75,
        ɛ::Real=0, epsilon::Real=ɛ,
        # num_tournament_groups = 20,
        selection::T1=tournament(cld(populationSize, 20), select=argmax),
        crossover::T2=TPX,
        mutation::T3=PLM(1.0; pm = 0.75, η = 2),
        metrics = ConvergenceMetric[AbsDiff(Inf)],
        optsys::T5) where {T1, T2, T3, T5} =
        new{T1,T2,T3,typeof(metrics),T5}(populationSize, crossoverRate, mutationRate, epsilon, selection, crossover, mutation, metrics, optsys)
end
population_size(method::QD) = method.populationSize
default_options(method::QD) = (abstol=Inf, reltol=Inf, successive_f_tol = 4, iterations=5, parallelization = Val(:thread), show_trace=true, show_every=1, store_trace=true,)
summary(m::QD) = "QD[P=$(m.populationSize),x=$(m.crossoverRate),μ=$(m.mutationRate),ɛ=$(m.ɛ)]"
show(io::IO,m::QD) = print(io, summary(m))
ismultiobjective(obj::EvolutionaryObjective{<:Any,<:Any,<:Any,<:Val}) = false


"""QD state type that captures additional data from the objective function in `valarray`"""
mutable struct QDState{T} <: AbstractOptimizerState 
    fittestValue::Float64  #* fitness of the fittest individual
    fittestChromosome::T  #* fittest chromosome (vector of gene values)
    objective_values::Matrix{Float64} #* array to store fitness, period, and amplitude of the population
end  
value(s::QDState) = s.fittestValue #return the fitness of the fittest individual
minimizer(s::QDState) = s.fittestChromosome #return the fittest individual
get_objective_values(s::QDState) = s.objective_values #return the objective values of the population


"""Get the fitness values from the objective_values matrix. View of the first row."""
function get_fitness(objective_values::AbstractMatrix)
    return @view objective_values[1, :]
end

"""Get the periods from the objective_values matrix. View of the second row."""
function get_periods(objective_values::AbstractMatrix)
    return @view objective_values[2, :]
end

"""Get the amplitudes from the objective_values matrix. View of the third row."""
function get_amplitudes(objective_values::AbstractMatrix)
    return @view objective_values[3, :]
end



"""Initialization of my custom QD algorithm state that captures additional data from the objective function\n
    - `method` is the QD method\n
    - `options` is the options dictionary\n
    - `objfun` is the objective function\n
    - `population` is the initial population
"""
function initial_state(method::QD, options, objfun, population) 

    #- Initialize the main output array
    objective_values = zeros((3, method.populationSize))
    fitvals = get_fitness(objective_values)

    #- Evaluate population fitness, period and amplitude
    # population_matrix = parent(parent(population))
    value!(objfun, objective_values, population)

    #- Get the maximum fitness and index of the fittest individual
    maxfit, maxfitidx = findmax(fitvals)

    #- Initialize the state object
    return QDState(maxfit, copy(population[maxfitidx]), objective_values)
end

"""Update state function that captures additional data from the objective function"""
function update_state!(objfun, constraints, state::QDState, parents, method::QD, options, itr)
    populationSize = method.populationSize
    rng = options.rng
    offspring = deepcopy(parents) 

    fitvals = get_fitness(state.objective_values)


    #* select offspring via tournament selection
    selected = method.selection(fitvals, populationSize, rng=rng)

    #* perform mating with TPX
    recombine!(offspring, parents, selected, method, rng=rng)

    #* perform mutation with BGA
    mutate!(offspring, method, constraints, rng=rng) #* only mutate descendants of the selected

    #* calculate fitness, period, and amplitude of the population
    evaluate!(objfun, state.objective_values, offspring, constraints)

    #* select the best individual
    maxfit, maxfitidx = findmax(fitvals)
    state.fittestChromosome .= offspring[maxfitidx]
    state.fittestValue = maxfit

    #* replace population
    parents .= offspring

    return false
end


# function recombine!(offspring, parents, selected, method::QD;
#                     rng::AbstractRNG=default_rng())
#     n = length(selected)
#     mates = ((i,i == n ? i-1 : i+1) for i in 1:2:n)
#     for (i,j) in mates
#         p1, p2 = parents[selected[i]], parents[selected[j]]
#         if rand(rng) < method.crossoverRate
#             offspring[i], offspring[j] = method.crossover(p1, p2, rng=rng)
#         else
#             offspring[i], offspring[j] = p1, p2
#         end
#     end
# end


"""
    recombine!(offspring, parents, selected, method::QD; rng::AbstractRNG=default_rng())

Perform recombination (crossover) on the selected individuals from the parent population
to create the offspring population.

# Arguments
- `offspring`: The array to store the resulting offspring.
- `parents`: The array containing the parent population.
- `selected`: Indices of selected individuals for mating.
- `method::QD`: The Quality Diversity method object containing algorithm parameters.
- `rng`: Random number generator (optional, default is `default_rng()`).
"""
function recombine!(offspring, parents, selected, method::QD;
                    rng::AbstractRNG=default_rng())
    n = length(selected)
    
    # Create pairs for mating using non-circular pairing
    # If n is odd, the last individual is paired with the previous one
    mates = ((i, i == n ? i-1 : i+1) for i in 1:2:n)
    
    # Pre-determine crossover events for efficiency
    # This creates a boolean array indicating which pairs will undergo crossover
    crossover_events = rand(rng, n÷2) .< method.crossoverRate

    # Iterate through the mating pairs
    for (idx, (i, j)) in enumerate(mates)
        p1, p2 = parents[selected[i]], parents[selected[j]]
        
        if crossover_events[idx]
            # Perform crossover
            o1, o2 = method.crossover(p1, p2, rng=rng)
            offspring[i] .= o1
            offspring[j] .= o2
        else
            # Direct copy if no crossover
            offspring[i] .= p1
            offspring[j] .= p2
        end
    end
end

function mutate!(population, method::QD, constraints;
                 rng::AbstractRNG=default_rng())
    n = length(population)
    for i in 1:n
        if rand(rng) < method.mutationRate
            method.mutation(parent(population[i]), rng=rng)
        end
        population[i] .= abs.(population[i])
        apply!(constraints, population[i])
    end
end



function evaluate!(objfun, objective_values, population, constraints::WorstFitnessConstraints)
    # calculate fitness of the population
    value!(objfun, objective_values, population)
    # apply penalty to fitness
    penalty!(get_fitness(objective_values), constraints, population)
end










