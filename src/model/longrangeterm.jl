struct LongRangeTerm <: AbstractTerm{2}
    term::NTuple{2,Matrix{ComplexF64}}
    decay::ComplexF64
    coeff::ComplexF64
    LongRangTerm(term::NTuple{2,Matrix{ComplexF64}}, d, c) = new(term, d, c)
end

LongRangeTerm(term, λ, a=1) = newterm(LongRangeTerm, term, λ, a)
scaleterm(a, A::LongRangeTerm) = LongRangeTerm(A.term, a * A.coeff, A.decay)

ham(A, λ, B) = LongRangeTerm((A, B), λ)
