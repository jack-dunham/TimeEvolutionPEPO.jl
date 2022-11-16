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

    @tensoropt flu[b2; a2 3] := fl[b1; a1 1] * at[a1 4; a2] * m[1 2; 3 4] * (ab')[b2; b1 2]

    return LeftEnvTensor(flu)
end
function Base.:*(T::TransferMatrix, FR::RightEnvTensor)
    fr = totensor(FR)
    at = totensor(T.above)
    ab = totensor(T.below)
    m = totensor(T.middle)

    fru = similar(FR)

    @tensoropt fru[a1 1; b1] := at[a1 4; a2] * m[1 2; 3 4] * (ab')[b2; b1 2] * fr[a2 3; b2]

    return RightEnvTensor(fru)
end
