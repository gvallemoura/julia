function compute_Seis(C::AbstractPDMat,Vtm1::Array{Float64,2},d2::Float64,sqiv::Float64,EISmat::Array{Float64,2})

    value,V = eig(Vtm1)
    sS = C.chol[:L]*V*diagm(value.^d2)*sqiv
    S  = PDMat(sS*sS')
    EISmatS = (S*EISmat)'
    hu = -1/(1+trace(EISmatS))*S*EISmatS
    Seis = Symmetric(pdadd(hu,S))
    return Seis,S
end
