function lik_MSV_EIS(par::Array{Float64,2},yt::Array{Float64,2},Vini::Array{Float64,2},N::Int64,adj::Float64,semente::Int64)


# Parameters and constants
const LOG2PI = 1.83787706640934533908193770912475883960723876953125;
const LOG2   = 0.69314718055994528622676398299518041312694549560546875;
const LOGPI  = 1.1447298858494001638774761886452324688434600830078125;
const d      = par[1]
const d2     = d/2
const v      = par[2]
const T,K    = size(yt)
const Neis   = 200
const Arg    = v-K+1:v
const df_p1  = v+1
const inds   = find(tril!(ones(K,K)))
const Nb     = length(inds)            # number of estimated parameters from the covariance matrix or from the \Gamma matrix filled with (k^2+k)/2 auxiliary parameters
const ind_   = find(eye(K))            # index from a diagonal KxK matrix
const itmax  = 3;                      # maximum number of iterations;
const tol    = 0.01;                   # EIS tolerance to stop iterations;
const iv     = 1/v
const sqiv   = sqrt(iv)
const Ones   = ones(Neis,1)
indd         = Array{Int64,1}(K+1)
for i=1:K
    hu      = find(inds.==ind_[i])
    indd[i] = hu[1]                    # indicate the linear indice in "inds" which corresponds to a diagonal parameter;
end
indd[end] = Nb+1
cC        = zeros(K,K)
cC[inds]  = par[3:Nb+2]                # maps parameters to cC matrix in order to recriate matrix cC;
cC        = reshape(cC,K,K)            # reorganize matrix;
C         = PDMat(cC*cC')


# Parameters for EIS iterations
it      = 0
diff    = 100                         # initial difference;
bet     = zeros(Nb+1,T)              # initialize vector to save EIS parameters;
betas_0 = bet                         # initialize EIS parameters at zero for initial sampler. See paragraph bellow eq. (17);
VINI    = Array{Float64,3}(K,K,Neis)
for i=1:Neis
    VINI[:,:,i]    = Vini             # state at t=0. Initial condition is treated as given
end
Y       = Array{Float64,2}(Neis,T-1)    # dependent variable for EIS regression;
XX      = Array{Float64,3}(Neis,Nb+1,T) # set space for Neis draws of EIS parameters at each period t;
Ytm1    = Array{Float64,2}(Neis,1)      # set space for Neis evaluations of the EIS objective function.



# Begin EIS iterations.
while it<itmax
    # Use CRNs
    srand(semente)
    # Initialize EIS sampler. See eq.(17):
    V0        = copy(VINI)               # initial state is given in this exercise;
    df_eis    = bet[end,:].+df_p1
    df_eis    = [df_eis; df_p1]          # degrees of freedom of EIS samplers from t=1 to t=T. Note that for t=T, df_eis=df_p1, since filter estimate is equal to smoothed estimate at T;
    for t=1:T
        BetMat  = zeros(K,K)                   # BetMat will be used as the \Gamma matrix from Eq. (17), which collects (k^2+k)/2 auxiliary parameters from the scale matrix of the EIS sampler
        BetMat[inds] = bet[1:Nb,t]
        BetMat       = Symmetric(BetMat,:L)#BetMat+tril(BetMat,-1)'
        EISmat       = BetMat+yt[t,:]*yt[t,:]'
        df_eis_t     = df_eis[t];
        if t==1
            cVd = Vini.^(d2)
            # Veist = Array{Float64,3}(K,K,Neis)
            # XX   = Array{Float64,2}(Neis,Nb+1)
            sS  = C.chol[:L]*cVd*sqiv
            # Compute symmetric PDMatrix scale matrix
            S   = (sS*sS')
            # Ss  = PDMat(S)
            EISmatS = EISmat*S#(Ss*EISmat)'
            Seis = Symmetric(S-1/(1+trace(EISmatS))*S*EISmatS)
            if isposdef(Seis)
                Seis = PDMat(Seis)
            else
                Seis = nearestSPD(Seis)
            end
            Veist = rand(Wishart(df_eis_t,Seis),Neis)
            for  i=1:Neis
                XX[i,1,t] = Veist[i][1,1]*0.5
                XX[i,2,t] = Veist[i][2,1]
                XX[i,3,t] = Veist[i][3,1]
                XX[i,4,t] = Veist[i][2,2]*0.5
                XX[i,5,t] = Veist[i][3,2]
                XX[i,6,t] = Veist[i][3,3]*0.5
                XX[i,7,t] = logdet(Veist[i])*0.5
                V0[:,:,i] = Veist[i]
            end
        else
            Veist = Array{Float64,3}(K,K,Neis)
            # XX    = Array{Float64,2}(Neis,Nb+1)
            for i=1:Neis
                # Compute Eigenvalue decomposition
                value,V = eig(V0[:,:,i])
                sS  = C.chol[:L]*V*diagm(value.^(d2))*sqiv
                # Compute symmetric PDMatrix scale matrix
                S   = (sS*sS')
                # Ss  = PDMat(S)
                EISmatS = EISmat*S#(Ss*EISmat)'
                Seis = Symmetric(S-1/(1+trace(EISmatS))*S*EISmatS)
                # Seis2,S2 = compute_Seis(C,V0[:,:,i],d2,sqiv,EISmat)
                # println(Seis-Seis2)
                # println('\n')
                # println(pdadd(S,S2))

                if isposdef(Seis)
                    Seis = PDMat(Seis)
                else
                    Seis = nearestSPD(Seis)
                end
                # Compute integrating constant \Chi_{t} for smoothing. See Eq. (18).
                logdetStm1 = logdet(Seis)
                logdetS    = logdet(S);
                Ytm1[i]    = 0.5*(df_eis_t*logdetStm1-v*logdetS); # log of integrating constant, which is a scalar;
                Veist[:,:,i] = rand(Wishart(df_eis_t,Seis))
                XX[i,1,t] = Veist[1,1,i]*0.5
                XX[i,2,t] = Veist[2,1,i]
                XX[i,3,t] = Veist[3,1,i]
                XX[i,4,t] = Veist[2,2,i]*0.5
                XX[i,5,t] = Veist[3,2,i]
                XX[i,6,t] = Veist[3,3,i]*0.5
                XX[i,7,t] = logdet(Veist[:,:,i])*0.5
            end
            Y[:,t-1] = Ytm1;
            if t<T
                V0 = Veist
            end
        end
    end
    # Backward EIS loop to smooth EIS estimates of t (\Chi_{t}) with integrating constant of t+1 (\Chi_{t+1})
    for t = T-1:-1:1
        X = [Ones XX[:,:,t]]  # EIS regressors. Obs: X uses the sample of N draws to estime EIS parameters, collected by beta;
        b = (X'*X)\(X'*Y[:,t])     # EIS parameters. Beta is a Nb+2 x 1 matrix;
        bet[:,t] = b[2:end]        # Save EIS parameters eliminating those w.r.t. the constant. Matrix bet(:,t) becomes Nb+1 x 1 again;
    end
    # Check convergence:
    diff = sum(sum(abs.(bet[1:end-1,:]-betas_0[1:end-1,:]))) # compute sum of individual relative difference between beta_{t} and beta_{t-1}. Obs: it is also possible to compute percentage change in parameters;
    #println(diff)
    betas_0 = copy(bet)
    # Count EIS iterations
    it = it+1
end
# Sample from optimized EIS sampler:
V0   = Array{Float64,3}(K,K,N)
for i=1:N
    V0[:,:,i] = Vini              # state at t=0. Initial condition is treated as given
end
Veist  = Array{Float64,3}(K,K,N);
df_eis = bet[end,:].+df_p1
df_eis = [df_eis; df_p1]          # degrees of freedom of EIS samplers from t=1 to t=T. Note that for t=T, df_eis=df_p1, since filter estimate is equal to smoothed estimate at T;
srand(semente*3)                  # use different seed after optimization

# Evaluate EIS Integrand for MC integration using draws from optimized EIS sampler.
ratio  = Array{Float64,2}(T,N);
BetMat = zeros(K,K)              # BetMat will be used as the \Gamma matrix from Eq. (17), which collects (k^2+k)/2 auxiliary parameters from the scale matrix of the EIS sampler

# Construct moments of EIS optimized sampler
for t=1:T
#    ytt          = yt[t,:]
    BetMat[inds] = bet[1:Nb,t]
    BetMatF      = Symmetric(BetMat,:L)#BetMat+tril(BetMat,-1)'
    EISmat       = BetMatF+yt[t,:]*yt[t,:]'
    df_eis_t     = df_eis[t];
    ArgEIS       = (df_eis_t-K+1:df_eis_t)
    hu        = 1
    for i=1:N
        # Compute Eigenvalue decomposition
        value,V = eig(V0[:,:,i])
        if minimum(value)<0
            println(t,i)
            println(value)
            println("Trocando auto valor")
            println('\n')
            # value[1]=eps(Float64)
        end
        sS  = C.chol[:L]*V*diagm(value.^(d2))*sqiv
        # Compute symmetric PDMatrix scale matrix
        S   = (sS*sS')
        # Ss  = PDMat(S)
        EISmatS = EISmat*S#(Ss*EISmat)'
        Seis = Symmetric(S-1/(1+trace(EISmatS))*S*EISmatS)
        if isposdef(Seis)
            Seis = PDMat(Seis)
        else
            println(t,i)
            Seis = nearestSPD(Seis)
            # println('\n')
            # println(exp(logdet(Seis)))
            hu=69
        end
        Veist[:,:,i] = rand(Wishart(df_eis_t,Seis))
        Veisti = PDMat(Veist[:,:,i])
        logdetVeis   = logdet(Veisti)
        if hu==69
            println(logdetVeis)
            valor,Vetor = eig(Veist[:,:,i])
            println(valor)
            println('\n')
            hu=1
        end
        # Measurement density in logs
        lgt = -K/2*LOG2PI+0.5*logdetVeis-0.5*quad(Veisti,yt[t,:])#yt[t,:]'*Veist[:,:,i]*yt[t,:]
        # State Transition Density in logs
        cnstt  = logdet(S)*(-v/2)-(v*K*.5)*LOG2-((K*(K-1))/4)*LOGPI-sum(lgamma.(Arg/2))
        kernel = logdetVeis*((v-K-1)/2) -.5*sum(diag(S\Veist[:,:,i]))
        lpt =  cnstt+kernel
        # Importance Sampler in logs
        cnstt  = logdet(Seis)*(-df_eis_t/2)-(df_eis_t*K*.5)*LOG2-((K*(K-1))/4)*LOGPI-sum(lgamma.(ArgEIS/2))
        kernel = logdetVeis*((df_eis_t-K-1)/2) -.5*sum(diag(Seis\Veist[:,:,i]))
        lmt    = cnstt+kernel
        # IS ratio
        ratio[t,i] = (lgt+lpt-lmt)
    end
    if t<T
        V0 = Veist
    end
end

lik = mean(exp.(sum(ratio.-adj,1)))    # adjust ratio against overflows due to exponentiation;
loglik = log(lik)+T*adj                # loglikelihood;
n_loglik = -loglik

return n_loglik

end
