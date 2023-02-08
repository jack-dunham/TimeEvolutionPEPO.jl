const AbstractMatrixProductOperator{L,M<:AbsTen{2,2}} = AbstractOnLattice{MPO,L,M}
const AbstractMPO = AbstractMatrixProductOperator

const MatrixProductOperator{L<:AbstractLattice,M<:AbsTen{2,2}} = OnLattice{MPO,L,M}

const InfiniteMatrixProductOperator{Nx,Ny,M<:AbsTen{2,2}} = OnUnitCell{MPO,Nx,Ny,M}
const InfiniteMPO = InfiniteMatrixProductOperator

function MatrixProductOperator(l::AbstractLattice, m)
    return OnLattice(MPO(), l, m)
end

function InfiniteMatrixProductOperator{Nx,Ny}(m) where {Nx,Ny}
    return MatrixProductOperator(Infinite{Nx,Ny}(), m)
end
