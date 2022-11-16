function ntu_metric(A,B,Qa,Qb)
end

# calculate g_a
function ntu_ga(g, Mb)
    @tensor ga[1 2 -3; -4 7 8] := g[1 2 3 4; 5 6 7 8]*Mb[5 6; -4]*(Mb')[-3; 3 4]
end

# calculate g_a
function ntu_gb(g, Ma)
    p = (5,6,7,8,1,2,3,4)
    gb = _permute(g, p)
    return _permute(ntu_ga(gb,Ma), invperm(p))
end

# incomplete
function ntu_ja(g,Mb,Ra, Rb)
    @tensor ga[1 2 -3; -4 7 8] := g[1 2 3 4; 5 6 7 8]*Mb[5 6; -4]*(Mb')[-3; 3 4]
end
