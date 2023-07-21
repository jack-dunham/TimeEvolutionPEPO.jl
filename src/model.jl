abstract type AbstractModel end

# struct Model{N}
#     xdata::Array{ComplexF64,N}
#     ydata::Array{ComplexF64,N}
# end

struct ModelMPO <: AbstractModel
    A::Matrix{ComplexF64}
    B::Matrix{ComplexF64}
    C::Matrix{ComplexF64}
    D::Matrix{ComplexF64}
end

struct Model <: AbstractModel
    xdata::Matrix{ComplexF64}
    ydata::Matrix{ComplexF64}
    param::Vector{String}
    hamiltonian::String
    linops::Vector{String}
    Model(x, y, p, h, l) = new(x, y, p, h, l)
    Model(x, y) = new(x, y)
end

struct SimpleModel <: AbstractModel
    oneloc::Vector{Matrix{ComplexF64}}
    twoloc::Vector{Matrix{ComplexF64}}
    lin::Vector{Matrix{ComplexF64}}
end
SimpleModel() = SimpleModel([], [], [])

struct ModelData <: AbstractModel
    xdata::Array{ComplexF64,6}
    ydata::Array{ComplexF64,6}
end

macro model(ex...)
    rv = quote
        xd, yd = @infliouvillian $ex
        Model(xd, yd)
    end
    return rv
end

macro defmodel(rules_ex, equation_ex...)
    return defmodel(rules_ex, equation_ex...)
end

function defmodel(rules_ex::Expr, equation_ex...)
    rules_block = MacroTools.postwalk(rules_ex) do x
        @capture(x, A_ => B_) || return x
        return :($A = $(esc(B)))
    end
    rules_block.head = :block

    rules_tuple = MacroTools.postwalk(rules_ex) do x
        @capture(x, A_ => B_) || return x
        return :($(string(A)) * " = " * (string($(esc(B)))))
        # return :($(string(A)) => $(esc(B)))
    end
    rules_tuple.head = :vcat

    str_param = string.(rules_tuple.args)
    str_H = string(unblock(equation_ex[1]))
    str_Ls = [string.(unblock(equation_ex[2:end]))...]

    rv = quote
        let
            $rules_block
            data_x, data_y = @infliouvillian($(equation_ex[1]), $(equation_ex[2:end]...))
            Model(data_x, data_y, $rules_tuple, $str_H, $str_Ls)
        end
    end
    return rv
end
function Base.show(io::IO, model::Model)
    if all(x -> isdefined(model, x), fieldnames(Model))
        params_to_print = []

        ham = model.hamiltonian

        for param_str in model.param
            param = Meta.parse(param_str)

            if param.args[2] == 0
                rem = "(?>" * string(param.args[1]) * ")"
                re = "($(rem).*?)([\\+\\-](?![^\\[]*\\])|\$)\\s*"
                ham = replace(ham, Regex(re) => "")
            else
                push!(params_to_print, param_str)
            end
        end

        # Remove trailing +/-
        ham = replace(ham, r"[\+\-]\s*$" => "")

        println("Liouvillian data of type Matrix{ComplexF64} generated from:")
        println(io, "  Hamiltonian:")
        println("    ", ham)
        println(io, "  Lindblad operators: ")
        foreach(iv -> println("    ", iv[1], ". ", iv[2]), enumerate(model.linops))
        println("  where:")
        foreach(x -> println("    ", x), params_to_print)
    else
        println("Unknown model data of type Matrix{ComplexF64}:")
        show(io, model.xdata)
        show(io, model.ydata)
    end
    return nothing
end

function lrtno()
    # 1 -> do nothing 2-> conventional path 3-> unconventional path 4-> fin
    # (L,D) -> (R,U)
    # -----------
    # 1,1,1,1 = I
    # 1,2,1,2 = I
    # 2 1 2 4 = I
    # 1 4 1 4 = I
    # 4 1 4 4 = I
    # -----------
    # 1 1 1 2 = A
    # 1 1 2 4 = A
    # 1 2 2 4 = I
    # 1 2 4 4 = B
    # 2 1 4 4 = B
    # -----------
    # 3 1 3 4 = I
    # 3 2 4 4 = I
    # 1 1 3 4 = B
    # -----------
    # TE (L,D) -> (R,U)
    # -----------
    # 1,1,1,1 = I
    # 1,2,1,2 = I
    # 2 1 2 4 = I
    # 1 4 1 4 = I
    # 4 1 4 4 = I
    # -----------
    # 1 1 1 2 = A
    # 1 1 2 4 = A
    # 1 2 2 4 = I
    # 1 2 4 4 = B
    # 2 1 4 4 = B
    # -----------
    # 3 1 3 4 = I
    # 3 2 4 4 = I
    # 1 1 3 4 = B
    # -----------
end
