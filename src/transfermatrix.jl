struct TransferMatrix
    above::MPSTensor
    middle::MPOTensor
    below::MPSTensor
end

# do this to get other FLs
function Base.:*(FL::LeftEnvTensor, T::TransferMatrix)
    fl = totensor(FL)
    at = totensor(T.above)
    ab = totensor(T.below)
    m = totensor(T.middle)

    flu = similar(FL)

    @tensoropt flu[b2; 3 a2] := fl[b1; 1 a1] * at[4 a1; a2] * m[1 2; 3 4] * (ab')[b2; 2 b1]

    return LeftEnvTensor(flu)
end
function Base.:*(T::TransferMatrix, FR::RightEnvTensor)
    fr = totensor(FR)
    at = totensor(T.above)
    ab = totensor(T.below)
    m = totensor(T.middle)

    fru = similar(FR)

    @tensoropt fru[1 a1; b1] := at[4 a1; a2] * m[1 2; 3 4] * (ab')[b2; 2 b1] * fr[3 a2; b2]

    return RightEnvTensor(fru)
end
