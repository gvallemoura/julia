function gen_data_Wishart(K::Int64,T::Int64,d::Float64,v::Int64,semente::Int64,V0::Array{Float64,2},C::AbstractPDMat)
    # Declare variables
    Vtrue = Array{Float64}(K,K,T)
    yt    = Array{Float64}(T,K)
    # Fix seed
    srand(semente)
    # Loop over time periods
    for t = 1:T
        # Compute Eigenvalue decomposition
        value,V = eig(V0)
        cV0 = V*diagm(value).^(d/2)
        sS  = C.chol[:L]*cV0*(iv/2)
        # Compute symmetric PDMatrix scale matrix
        S   = PDMat(sS*sS')
        # Generate and save unobservable covariance
        Vtrue[:,:,t] = rand(Wishart(v,S))
        V1 = PDMat(Vtrue[:,:,t])
        # Generate and save observable data
        yt[t,:] = rand(MvNormal(inv(V1)))
        V0 = Vtrue[:,:,t]
    end
    return yt, Vtrue
end
