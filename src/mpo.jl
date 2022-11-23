const AbstractMatrixProductOperator{L,M<:AbsTen{2,2}} = AbstractLattice{MPO,L,M}
const AbstractMPO = AbstractMatrixProductOperator

const MatrixProductOperator{L<:LatticeStyle,M<:AbsTen{2,2}} = Lattice{MPO,L,M}

const InfiniteMatrixProductOperator{Nx,Ny,M<:AbsTen{2,2}} = UnitCell{MPO,Nx,Ny,M}
const InfiniteMPO = InfiniteMatrixProductOperator

function MatrixProductOperator(l::LatticeStyle, m)
    return Lattice(MPO(), l, m)
end

function InfiniteMatrixProductOperator{Nx,Ny}(m) where {Nx,Ny}
    return MatrixProductOperator(Infinite{Nx,Ny}(), m)
end
