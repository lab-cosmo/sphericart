
"""
`sizeY(maxL):`
Return the size of the set of spherical harmonics ``Y_l^m`` of
degree less than or equal to the given maximum degree `maxL`
"""
sizeY(maxL) = (maxL + 1)^2

"""
`lm2idx(l,m):`
Return the index into a flat array of real spherical harmonics ``Y_l^m``
for the given indices `(l,m)`. ``Y_l^m`` are stored in l-major order i.e.
```
	[Y(0,0), Y(1,-1), Y(1,0), Y(1,1), Y(2,-2), ...]
```
"""
lm2idx(l::Integer, m::Integer) = m + l + (l*l) + 1

"""
Inverse of `lm2idx`: given an index into a vector of Ylm values, return the 
`(l, m)` indices.
"""
function idx2lm(i::Integer) 
	l = floor(Int, sqrt(i-1) + 1e-10)
	m = i - (l + (l*l) + 1)
	return l, m 
end 


"""
Convenience wrapper around solid/spherical harmonics outputs. 
Store the basis as a vector but access via `(l, m)` pairs. 
"""
struct ZVec{TP} 
   parent::TP 
end 

getindex(Z::ZVec, i::Integer) = Z.parent[i]

getindex(Z::ZVec, l::Integer, m::Integer) = Z.parent[lm2idx(l, m)]
