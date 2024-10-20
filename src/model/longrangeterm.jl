struct LongRangeTerm <: AbstractTerm{2}
    term::NTuple{2,Matrix{ComplexF64}}
    decay::ComplexF64
    coeff::ComplexF64
end

# LongRangeTerm(term, λ, a=1) = newterm(LongRangeTerm, term, λ, a)
scaleterm(a, A::LongRangeTerm) = LongRangeTerm(A.term, A.decay, a * A.coeff)

# ham(A, λ, B) = LongRangeTerm((A, B), λ)
ham(A, λ, B) = newterm(LongRangeTerm, (A,B), λ, 1)
