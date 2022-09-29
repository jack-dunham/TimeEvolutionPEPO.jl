@doc raw"""
struct UnitCell{Nx,Ny,S<:IndexSpace}

Concrete struct to represent a tensor network unit cell with index space of type `S`. Has a
single field `label::SMatrix{Nx,Ny,String}` that allows one to define a label for each site
of the unit cell.
"""
struct UnitCell{Nx,Ny,S<:IndexSpace}
    label::SMatrix{Nx,Ny,String}
end

function UnitCell{Nx,Ny}(S::Type{<:IndexSpace}, l::Matrix{String}) where {Nx,Ny}
    return UnitCell{Nx,Ny,S}(SMatrix{Nx,Ny}(l))
end

function UnitCell{Nx,Ny}(S::Type{<:IndexSpace}) where {Nx,Ny}
    l = Matrix{String}(undef, Nx, Ny) 
    for x in 1:Nx, y in 1:Ny
        l[x,y] = string("x",x,"y",y)
    end
    return UnitCell{Nx,Ny}(S, l)
end

UnitCell{Nx,Ny,S}() where {Nx,Ny, S} = UnitCell{Nx,Ny}(S)

Base.size(::UnitCell{Nx,Ny}) where {Nx,Ny} = Nx,Ny
TensorKit.spacetype(::UnitCell{Nx,Ny,S}) where {Nx,Ny,S} = S

## over complicated
# function UnitCell(S::Type{<:IndexSpace}, l::Vararg{NTuple{Nx,Symbol}, Ny}) where {Nx,Ny}
#     sm = SMatrix{Nx,Ny,Symbol}(hcat(collect(collect.(l))...))
#     return UnitCell{Nx,Ny,S}(sm)
# end

const TrivialUnitCell{S<:IndexSpace} = UnitCell{1,1,S}
