module Folly

using KernelAbstractions: @kernel, @index, CPU, CUDA
export CPU, CUDA

export Outer, Inner, Block
export IndVar, Loop
export Native, KATarget

abstract type Kind end

# @index(Group, NTuple)[dim]

# Assumed to be parallel
struct Outer <: Kind end
struct Inner <: Kind end

# @index(Local, NTuple)[dim]
struct Block <: Kind end

# Recurrence relationship
# IndVar{Outer}(:gI)
# IndVar{Block}(:lI)
# IndVar{Dependent}(:I, ..., :(gI * N + lI))
# struct Dependent <: Kind end

# TODO: add metadata like unroll factor, 
# The induction variable is most often an index 
struct IndVar{Kind}
    name::Symbol
    kind::Kind
    recurrence::Expr
    lowerbound::Union{Number, Expr}
    upperbound::Union{Number, Expr}
end

# Represents an loop over a variable
# the body can contain statements and/or other loops
struct Loop
    indvar::Symbol
    body::Vector{Any}
end

# TODO:
# - From recurrence to affine-map
# - Dependent IndVars
# - Compilation to KernelAbstractions
#  - map IndVars to NDrange dimensions

abstract type Target end
struct KATarget <: Target
    device
    dims::Dict{IndVar, Int}
    KATarget(device) = new(device, Dict{IndVar, Int}())
end
struct Native <: Target end 

construct(::Target, expr::Expr, indvars) = expr
function construct(target::Target, loop::Loop, indvars)
    idx = findfirst(indv->indv.name == loop.indvar, indvars)
    @assert idx !== nothing
    indvar = indvars[idx]
    construct(target, indvar, loop, indvars)
end 

function construct(target::Native, indvar::IndVar, loop::Loop, indvars)
    body = map(expr->construct(target, expr, indvars), loop.body)
    name = indvar.name
    quote
        $name = $(indvar.lowerbound)
        while $name <= $(indvar.upperbound)
            $(body...)
            $name = $(indvar.recurrence)
        end
    end
end

function construct(target::KATarget, indvar::IndVar{Inner}, loop::Loop, indvars)
    body = map(expr->construct(target, expr, indvars), loop.body)
    name = indvar.name
    quote
        $name = $(indvar.lowerbound)
        while $name <= $(indvar.upperbound)
            $(body...)
        end
    end
end

index_expr(::IndVar{Outer}, dim) = :(@index(Group, NTuple)[$dim])
index_expr(::IndVar{Block}, dim) = :(@index(Local, NTuple)[$dim])

function construct(target::KATarget, indvar::IndVar, loop::Loop, indvars)
    body = map(expr->construct(target, expr, indvars), loop.body)
    name = indvar.name
    dim = target.dims[indvar]
    quote
        $name = $(index_expr(indvar, dim))
        if $(indvar.lowerbound) <= $name <= $(indvar.upperbound)
            $(body...)
        end
    end
end


function compile(target::Native, loop::Loop, indvars, freevars)
    body = construct(target, loop, indvars)
    expr = quote
        function $(gensym(:folly))($(freevars...))
            $(body)
        end
    end
    eval(expr)
end

# TODO: Sort indvars according to depth
function compile(target::KATarget, loop::Loop, indvars, freevars)
    outer_indvars = filter(iv -> iv.kind isa Outer, indvars)
    block_indvars = filter(iv -> iv.kind isa Block, indvars)

    @assert length(block_indvars) == 0
    dim = 1
    for iv in block_indvars
        target.dims[iv] = dim
        dim += 1
    end
    for iv in outer_indvars
        target.dims[iv] = dim
        dim += 1
    end
    body = construct(target, loop, indvars)

    workgroupsize = append!(
                        map(iv->iv.upperbound, block_indvars),
                        map(iv->1, outer_indvars))
    ndrange       = append!(
                        map(iv->iv.upperbound, block_indvars),
                        map(iv->iv.upperbound, outer_indvars))
    ka_func = quote 
    end

    expr = quote
        function $(gensym(:folly))($(freevars...); dependencies=nothing)
            @kernel function kernel($(freevars...))
                $(body)
            end
            workgroupsize = ($(workgroupsize...),)
            ndrange       = ($(ndrange...),)

            k = kernel($(target.device), workgroupsize)
            k($(freevars...), ndrange=ndrange, dependencies=dependencies)
        end

    end
    eval(expr)
end

function freevars(loop::Loop, indvars::Set{Symbol} = Set{Symbol}(), free::Set{Symbol} = Set{Symbol}())
    @assert loop.indvar ∉ indvars
    push!(indvars, loop.indvar)
    for expr in loop.body
        freevars(expr, indvars, free)
    end
    delete!(indvars, loop.indvar)
    return free
end

function freevars(name::Symbol, indvars::Set{Symbol}, free::Set{Symbol})
    if name ∉ indvars
        push!(free, name)
    end
    return free
end

function freevars(expr::Expr, indvars::Set{Symbol}, free::Set{Symbol})
    if expr.head === :ref || expr.head === :(=) || expr.head === :(+=)
        for arg in expr.args
            freevars(arg, indvars, free)
        end 
    elseif expr.head === :call
        # skip first arg
        for i in 2:length(expr.args)
            freevars(expr.args[i], indvars, free)
        end
    else
        @warn "Unable to handle expr" expr.head
    end
    return free
end

# Very simple fusion, we expect that the indvars match
# this does not check that the loops can be fused.
# i.e. reductions and cross loop dependencies can be broken
function fuse!(loop::Loop, other::Loop)
    if loop.indvar == other.indvar
        for expr in other.body
            if expr isa Loop
                n = expr.name
                idxs = findall(e -> e isa Loop && e.name == n, loop.body)
                if length(idxs) == 1
                    fuse(loop.body[idxs[1]], expr)
                    continue
                elseif length(idxs) > 1
                    error("IndVar not unique on loop-level")
                end 
            end
            push!(loop.body, expr)
        end
    end
    return nothing
end

end # module
