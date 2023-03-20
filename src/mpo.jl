const AbstractMatrixProductOperator{L,M<:AbsTen{2,2}} = AbstractOnLattice{L,M}
const AbstractMPO = AbstractMatrixProductOperator

const MatrixProductOperator{L<:AbstractLattice,M<:AbsTen{2,2}} = OnLattice{L,M}

const InfiniteMatrixProductOperator{Nx,Ny,M<:AbsTen{2,2}} = OnUnitCell{Nx,Ny,M}
const InfiniteMPO = InfiniteMatrixProductOperator
