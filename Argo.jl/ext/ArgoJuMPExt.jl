module ArgoJuMPExt

import Argo
import JuMP

"""
    Argo.minimize(model::JuMP.GenericModel; name=:x)

Convert the objective of the JuMP `model` into an Argo `Problem`; see
`Argo.minimize(::MOI.ModelLike)`.
"""
function Argo.minimize(model::JuMP.GenericModel; name::Symbol=:x)
    return Argo.minimize(JuMP.backend(model); name)
end

"""
    Argo.term(f::JuMP.AbstractJuMPScalar, x::AbstractVector; name=:x)

Convert the JuMP expression `f` into an Argo objective term over a single
vector variable `name` whose `i`th coordinate is `x[i]`.
"""
function Argo.term(
    f::JuMP.AbstractJuMPScalar,
    x::AbstractVector{<:JuMP.AbstractVariableRef};
    name::Symbol=:x,
)
    return Argo.term(JuMP.moi_function(f), JuMP.index.(vec(x)); name)
end

end
