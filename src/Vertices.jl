struct Vertex{T, U<:Number}
  position::AbstractVector{T}
  value::U
end
Vertex(x::AbstractVector{T}, f::F) where {T, F} = Vertex(x, f(x))

value(v::Vertex) = v.value
position(v::Vertex) = v.position
newposition(a, ϵ, b) = a + ϵ .* (a - b)


function vertexpositions(ic::AbstractVector{T}, initial_steps::AbstractVector{U}
    ) where {T<:Number, U<:Number}
  if any(iszero.(initial_steps))
    throw(ArgumentError("initial_steps, $initial_steps  must not have any zero
                        values"))
  end
  if length(ic) != length(initial_steps)
    throw(ArgumentError("ic, $ic must be same length as initial_steps
                        $initial_steps"))
  end
  dim = length(ic)
  positions = Vector{Vector{promote_type(T,U)}}()
  for i ∈ 1:dim+1
    x = [ic[j] + ((j == i) ? initial_steps[j] : zero(U)) for j ∈ 1:dim]
    push!(positions, x)
  end
  return positions
end


# must explicitly use <= and >= because == can't overridden and will
# be used in conjunction with < to create a <=
import Base: isless, +, -, <=, >=, isequal, isnan, hash
Base.isless(a::Vertex, b::Vertex) = abs(value(a)) < abs(value(b))
Base.:<=(a::Vertex, b::Vertex) = abs(value(a)) <= abs(value(b))
Base.:>=(a::Vertex, b::Vertex) = abs(value(a)) >= abs(value(b))
Base.:+(a::Vertex, b) = position(a) .+ b
Base.:-(a::Vertex, b::Vertex) = position(a) .- position(b)
#Base.:-(a, b::Vertex) = a .- position(b)
function Base.isequal(a::Vertex, b::Vertex)
  values_equal = value(a) == value(b) || (isnan(a) && isnan(b))
  positions_equal = all(position(a) .== position(b))
  return values_equal && positions_equal
end
#Base.isequal(a::Vertex, b::AbstractVector) = all(position(a) .== b)

Base.isnan(a::Vertex) = isnan(value(a))

function vertexsorter(x::Vertex{T, <:Complex}, y::Vertex{T, <:Complex}) where
  {T}
  xa, ya = angle(value(x)), angle(value(y))
  xa == ya || return isless(xa, ya)

  return vertexsorter_abs_position(x::Vertex, y::Vertex)
end

vertexsorter(x::Vertex, y::Vertex) = vertexsorter_abs_position(x, y)

function vertexsorter_abs_position(x::Vertex, y::Vertex)
  xv, yv = abs(value(x)), abs(value(y))
  xv == yv || return isless(xv, yv)

  xp, yp = position(x), position(y)
  return any(i->isless(xp[i], yp[i]), eachindex(xp))
end


