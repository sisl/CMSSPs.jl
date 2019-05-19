
## TYPES
"""
Compact structure for 2D point.
"""
struct Point
    x::Float64
    y::Float64
end
Point() = Point(0.0,0.0)
Point(v::V) where {V <: AbstractVector} = Point(v[1], v[2])
Point(t::Tuple{Float64,Float64}) = Point(t[1], t[2])

function point_dist(p1::Point, p2::Point, l::Real=2)
    s = SVector{2,Float64}(p1.x-p2.x, p1.y-p2.y)
    return norm(s,l)
end

function point_norm(p::Point, l::Real=2)
    s = SVector{2,Float64}(p.x,p.y)
    return norm(s,l)
end

function equal(p1::Point, p2::Point)
    return (isapprox(p1.x,p2.x) && isapprox(p1.y,p2.y))
end

function interpolate(p1::Point,p2::Point,frac::Float64)

    xval = p1.x + frac*(p2.x-p1.x)
    yval = p1.y + frac*(p2.y-p1.y)

    return Point(xval,yval)

end