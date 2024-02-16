function minimalcanonicalform!(tensor::AbsTen{2,4}; tol=1e-12, maxiter=10000)
    hor = id(domain(tensor)[1])
    ver = id(domain(tensor)[2])
    T = deepcopy(tensor)
    err = Inf
    numiter = 0

    elp = @elapsed begin
        for t in 1:maxiter
            numiter = t

            mcfgauge!(tensor, T, hor, ver)

            @tensoropt (x => 2, y => 2) p_h1[e1; e2] :=
                tensor'[e1 s w n; x y] * tensor[x y; e2 s w n]
            @tensoropt (x => 2, y => 2) p_h2[w1; w2] :=
                tensor'[e s w1 n; x y] * tensor[x y; e s w2 n]
            @tensoropt (x => 2, y => 2) p_v1[s1; s2] :=
                tensor'[e s1 w n; x y] * tensor[x y; e s2 w n]
            @tensoropt (x => 2, y => 2) p_v2[n1; n2] :=
                tensor'[e s w n1; x y] * tensor[x y; e s w n2]

            h_diff = p_h1 - transpose(p_h2)
            v_diff = p_v1 - transpose(p_v2)

            @tensor trketbra = tr(p_h1)
            err = 1 / real(trketbra) * (norm(h_diff)^2 + norm(v_diff)^2)

            # if err ≤ tol^2
            #     break
            # end

            if err ≤ tol
                break
            end
            hor = exp(-1 / 8 * 1 / trketbra * h_diff) * hor
            ver = exp(-1 / 8 * 1 / trketbra * v_diff) * ver
        end
    end
    @info "Minimal canonical form gauging: $err after $numiter iterations taking $(elp)s"
    return tensor
end

function mcfgauge!(p_out, p_in, g1, g2) #hor, ver
    g1i = permute(inv(g1), (2,), (1,))
    g2i = permute(inv(g2), (2,), (1,))
    @tensor p_out[k b; e s w n] =
        p_in[k b; ee ss ww nn] * g1[ee; e] * g2[ss; s] * g1i[ww; w] * g2i[nn; n]
    return p_out
end
