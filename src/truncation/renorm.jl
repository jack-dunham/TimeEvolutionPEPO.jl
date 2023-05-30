
# Case where a bond has been updated, and we wish to renormalize environment
function renormalize!(environment, pepo, old_bond, ::Val{1})

    len = size(pepo)[getaxis(old_bond)]

    # No need to renormalize when we only have one bond
    if len == 1
        return environment
    end

    site = old_bond.in
    sites = [site + i*site_diff for i in 0:(len + 1)]

    if reversed == false
        left = sites[begin:begin]
        right = sites[4:end]
    else
        left = sites[1:end-3]
        right = sites[end:end]
    end

    absorbleft!(environment, pepo, left)
    absorbright!(environment, pepo, right)

    return environment
end

function absorbright!(environment, pepo, right::Vector{CartesianIndex{2}})
    while !isempty(right)
        site_index = pop!(right)
        absorbright!(environment, pepo, site_index)
    end
    return environment
end
function absorbleft!(environment, pepo, left::Vector{CartesianIndex{2}})
    while !isempty(right)
        site_index = popfirst!(left)
        absorbleft!(environment, pepo, site_index)
    end
    return environment
end
