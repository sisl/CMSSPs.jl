# CMSSPs

Accompanying code repository for the paper 'Hybrid Planning for Dynamic Multimodal Stochastic Shortest Paths'. [ArXiv](https://arxiv.org/abs/1906.09094)


**N.B.** Due to legacy naming reasons, the attached code uses the acronyms `CMSSP` and `HHPC` instead of `DMSSP` and `HSP` respectively. Also, this code is primarily provided for illustrative purposes. There are several inter-related components, and pre-processing steps necessary to run the full planning framework.

## Setup
The CMSSPs repository is set up as a package with its own environment in [Julia 1.0](https://julialang.org/downloads/). Look at **Using someone else's project** at the Julia [package manager documentation](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1) for the basic idea. To get the code up and running (after having installed Julia), first `cd` into the `CMSSPs` folder.
Then start the Julia REPL and go into [package manager](https://julialang.github.io/Pkg.jl/v1/getting-started/) mode by pressing `]`, followed by:
```shell
(v1.0) pkg> activate .
(CMSSPs) pkg> instantiate
```
This will install the necessary dependencies and essentially reproduce the Julia environment required to make the package work. You can test this by exiting the package manager mode with the backspace key and then in the Julia REPL entering:
```shell
julia> using CMSSPs
```
The full package should then pre-compile. AFTER this step, you can start [IJulia](https://github.com/JuliaLang/IJulia.jl) (install it if you have not already) and navigate to the `notebooks/` folder:
```shell
julia> using IJulia
julia> notebook(dir="./notebooks/")
```
You can then run the notebook to get an idea of how to use the code for a specific domain. An overview of the code package itself is given below.


## Overview

The code consists of _domain-agnostic_ and _domain-specific_ components. The former defines the  
problem interface and the general HSP solution framework (with global and local layers) while the latter 
provides the necessary definitions to run the solver on a particular domain. 
In terms of Julia, the `CMSSPs` module has three submodules - `CMSSPModel` for formulating DMSSPs,
`HHPC` for the hybrid stochastic planning framework, and `CMSSPDomains` to define specific  problem domains.
At all levels, the code relies heavily on the [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl) framework for modeling and solving Markov Decision Processes. It also uses Julia's [parametric types](https://docs.julialang.org/en/v1/manual/types/index.html) for multiple dispatch.


### DMSSP Problem Formulation

The interface for defining a (CMSSP) DMSSP is in `src/models/cmssp_base.jl`. For those familiar with POMDPs.jl, a
DMSSP is defined as a concrete type of `POMDPs.MDP{S,A}`, with additional parameters for the context set. A number of other methods are included for extracting mode-switch and control actions, and defining the vertex type for the global layer.


### Hybrid Stochastic Planning

The HSP planning algorithm is defined in `src/hhpc/`, with each component implemented in a separate file.

- `src/hhpc/global_layer.jl` implements the open-loop planning of the global layer (the GlobalPlan procedure
in Algorithm 1
from the paper), through an _implicit_ A* search
based on [Graphs.jl](https://github.com/JuliaAttic/Graphs.jl).
- `src/hhpc/local_layer.jl` implements the LocalPreprocessing procedure from Algorithm 1 of the paper.
It has methods
for the terminal cost penalty, the finite-horizon value iteration, and others. Currently
we assume that the value iteration uses [local approximation](https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl), but other approximation schemes could be used.
- `src/hhpc/hhpc_framework.jl` implements the overall HSP framework from Algorithm 2 of the paper by using
the global and local layer code. It defines the HSP
framework as a `Solver` of POMDPs.jl. This solver requires the definition of a number of domain-dependent functions; see
the [specifying requirements](http://juliapomdp.github.io/POMDPs.jl/latest/specifying_requirements/)
page of POMDPs.jl for more on how that works.


## Usage

To use the code for solving problems _on a particular domain_, the various DMSSP problem components 
and the HSP solver requirements must be correctly implemented. Our experiments were run on the
Dynamic Real-time Multimodal Routing (DREAMR) domain, where an autonomous agent
has to be controlled under uncertainty from a start to a goal location
amidst a dynamic transit network in which it can use multiple modes of transportation. The `src/domains/dreamr/` folder contains
all the code pertaining to the DREAMR domain, such as the state and action spaces,
transition and reward models, subroutine implementations, and so on (much of it is derived
from the open-source [DreamrHHP.jl](https://github.com/sisl/DreamrHHP) repository for the original
paper).


### Problem Scenarios

We use the same scenarios as from the original DREAMR paper. Each scenario is encoded in a large JSON file with
route information for several hundreds of cars over 360 epochs, and there are 1000 such scenarios, so we do not attach the
files with the code. They can be obtained by running the [grid data generator](https://github.com/sisl/DreamrHHP/blob/master/data/grid_data_generator.jl) file from the original DREAMR repository, with `<min-cars> <max-cars>` set to `100 1000`.
Each episode is then generated as a dictionary embedded in a JSON file.


### Running the HSP Solution

The `scripts/dreamr/` folder has all the code for the top-level scripts to actually run the HSP solver
on DREAMR problems.
Before we can use HSP on a given instance, we need to run the local pre-processing step to compute
the local closed-loop policies. For DREAMR, we have separate policies for time-constrained movement
and time-unconstrained movement, and for the ride mode, the policy is effectively deterministic.
For the time-constrained and time-unconstrained movement, the scripts are
`dreamr_cf_policy.jl` and `dreamr_uf_policy.jl` respectively. Individual test scripts are
also provided to check
that the policies work. 

Finally, to run the overall HSP framework on DREAMR problems, the script `dreamr_simulator.jl` has been defined.
It takes as arguments the policies for constrained movement and unconstrained movement that were generated
by the local pre-processing.
The `dreamr_simulator.jl` script runs the HSP solver for DREAMR over a set of problems and reports
statistics for each episode - the total cost incurred, number of timesteps to be completed, and mode switches.
