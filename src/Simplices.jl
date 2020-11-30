module Simplices

using LinearAlgebra

include("Vertexes.jl")

function convergenceconfig(dim::Int, T::Type; kwargs...)
  kwargs = Dict(kwargs)
  xtol_abs = get(kwargs, :xtol_abs, zeros(real(T))) .* ones(Bool, dim)
  xtol_rel = get(kwargs, :xtol_rel, eps(real(T))) .* ones(Bool, dim)
  ftol_abs = get(kwargs, :ftol_abs, zero(real(T)))
  ftol_rel = get(kwargs, :ftol_rel, eps(real(T)))
  stopval = get(kwargs, :stopval, eps(real(T)))

  return (xtol_abs=xtol_abs, xtol_rel=xtol_rel,
          ftol_abs=ftol_abs, ftol_rel=ftol_rel,
          stopval=stopval)
end


struct Simplex{T<:Number, U<:Number}
  vertices::Vector{Vertex{T,U}}
  permabs::Vector{Int}
  function Simplex(vertices::Vector{Vertex{T,U}}) where {T<:Number, U<:Number}
    output = new{T,U}(vertices, zeros(Int64, length(vertices)))
    sort!(output)
    return output
  end
end

function Simplex(f::T, ic::AbstractVector{U}, step) where {T, U<:Number}
  return Simplex(f, vertexpositions(ic, step .* ones(Bool, length(ic))))
end

function Simplex(f::T, positions::U
    ) where {T, W<:Number, V<:AbstractVector{W}, U<:AbstractVector{V}}
  if length(unique(length.(positions))) != 1
    throw(ArgumentError("All entries in positions $positions must be the same
                        length"))
  end
  vertices = [Vertex(p, f(p)) for p in positions]
  return Simplex(vertices)
end

function partitionunitypositions(dims::Int)
  output = Vector{Vector{Vector{Bool}}}()
  for i ∈ CartesianIndices(Tuple(ones(Int64, dims) .* dims))
    v = Tuple(i)
    any(j->any(v[j] .== v[1:j-1]), 1:dims) && continue
    inner = Vector{Vector{Bool}}()
    push!(inner, zeros(Bool, dims))
    for j ∈ 2:dims
      x = zeros(Float64, dims)
      for k ∈ 1:(j-1)
        x[v[k]] = true
      end
      push!(inner, x)
    end
    push!(inner, ones(Bool, dims))
    @assert length(inner) == dims + 1
    push!(output, inner)
  end

  @assert length(output) == factorial(dims)
  return output
end

function Base.:∈(x, s::Simplex)
  return ∈(Vertex(x, one(value(first(s)))), s)
end
function Base.:∈(v::Vertex, s::Simplex)
  h = hash(s)
  function inner(i)
    swap!(s, i, v)
    x = hypervolume(s)
    swap!(s, v, i)
    return x
  end
  volume = mapreduce(inner, +, s.vertices)
  @assert h == hash(s) "The simplex has changed and it shouldn't have."
  return isapprox(volume, hypervolume(s), rtol=sqrt(eps()), atol=0)
end

import Base: length, iterate, push!, iterate, getindex
import Base: eachindex, sort!, hash, extrema
Base.length(s::Simplex) = length(s.vertices)
function Base.push!(s::Simplex, v::Vertex)
  push!(s.vertices, v)
  l = length(s.vertices)
  return nothing
end
Base.iterate(s::Simplex) = iterate(s.vertices)
Base.iterate(s::Simplex, counter) = iterate(s.vertices, counter)
Base.getindex(s::Simplex, index) = s.vertices[index]

function Base.sort!(s::Simplex)
  sort!(s.vertices, lt=vertexsorter)
  sortperm!(s.permabs, s.vertices, by=x->abs(value(x)))
  return nothing
end

function Base.extrema(s::Simplex)
  return [extrema(position(v)[i] for v in s) for i in 1:dimensionality(s)]
end

dimensionality(s::Simplex) = length(s) - 1

remove!(s::Simplex, v::Vertex) = filter!(x -> !isequal(x, v), s.vertices)
remove!(s::Simplex, x::Vector) = deleteat!(s.vertices, x)

issortedbyangle(s::Simplex) = issorted(s, by=v->angle(value(v)))

selectabs(s, index) = s.vertices[s.permabs[index]]

bestvertex(s::Simplex) = selectabs(s, 1)
worstvertex(s::Simplex) = selectabs(s, length(s))
secondworstvertex(s::Simplex) = selectabs(s, length(s) - 1)

function centroidposition(s::Simplex, ignoredvertex=worstvertex(s))
  g(v) = isequal(v, ignoredvertex) ? zero(position(v)) : position(v)
  return mapreduce(g, +, s) / (length(s) - 1)
end

centre(s::Simplex) = mapreduce(v->position(v), +, s) / length(s)

hypervolume(s::Simplex) = hypervolume(s.vertices)

function hypervolume(vs::AbstractVector{V}) where {V<:Vertex}
  m = hcat((vcat(position(v), 1) for v in vs)...)
  d = length(position(vs[1]))
  return abs(det(m)) / factorial(d)
end

function swap!(s::Simplex, this::Vertex, forthat::Vertex)
  before = deepcopy(s)
  @assert this ∈ s.vertices
  lengthbefore = length(s)
  remove!(s, this)
  @assert length(s) == lengthbefore - 1 "$(length(s)), $lengthbefore"
  push!(s, forthat)
  @assert forthat ∈ s.vertices
  sort!(s)
  @assert length(s) == lengthbefore
  return nothing
end

swapworst!(s::Simplex, forthis::Vertex) = swap!(s, worstvertex(s), forthis)

function closestomiddlevertex(s::Simplex)
  mid = mapreduce(position, +, s) ./ length(s)
  _, index = findmin(map(v->sum((position(v) - mid).^2), s))
  return s[index]
end

function assessconvergence(simplex, config)

  if abs(value(bestvertex(simplex))) <= config[:stopval]
    return :STOPVAL_REACHED
  end

  toprocess = Set{Int}(1)
  processed = Set{Int}()
  while !isempty(toprocess)
    vi = pop!(toprocess)
    v = simplex.vertices[vi]
    connectedto = Set{Int}()
    for (qi, q) ∈ enumerate(simplex)
      thisxtol = true
      for (i, (pv, pq)) ∈ enumerate(zip(position(v), position(q)))
        thisxtol &= isapprox(pv, pq, rtol=config[:xtol_rel][i],
                                     atol=config[:xtol_abs][i])
      end
      thisxtol && push!(connectedto, qi)
      thisxtol && for i in connectedto if i ∉ processed push!(toprocess, i) end end
    end
    push!(processed, vi)
  end
  allxtol = all(i ∈ processed for i ∈ 1:length(simplex))
  allxtol && return :XTOL_REACHED

  allftol = true
  for (vi, v) ∈ enumerate(simplex)
    for qi ∈ vi+1:length(simplex)
      q = simplex.vertices[qi]
      allftol &= all(isapprox(value(v), value(q),
                              rtol=config[:ftol_rel], atol=config[:ftol_abs]))
      all(position(v) .== position(q)) && return :XTOL_DEGENERATE_SIMPLEX
    end
  end
  allftol && return :FTOL_REACHED

  return :CONTINUE
end

function _πtoπ(ϕ)
  ϕ < -π && return _πtoπ(ϕ + 2π)
  ϕ > π && return _πtoπ(ϕ - 2π)
  return ϕ
end

windingangle(s::Simplex{T, <:Real}) where {T} = zero(T)
function windingangle(s::Simplex{T, <:Complex}) where {T}
  return sum(_πtoπ.(angle.(value.(circshift(s.vertices, -1))) .-
                    angle.(value.(s.vertices))))
end

windingnumber(s::Simplex{T, <:Real}) where {T} = 0
function windingnumber(s::Simplex{T, <:Complex}) where {T}
  radians = windingangle(s)
  return isfinite(radians) ? Int64(round(radians / 2π)) : Int64(0)
end

end
