function lik_KChi(par::Array{Float},yt::Array{Float64,1},Vini::Array{Float64,2},N::Int64,adj::Float64,semente::Int64)

# Computes the negative of the loglikelihood of the Wishart MSV model:
#Densidade de observáveis:
#$$p(y_t|\Sigma_t)\sim N(0,\Sigma_t)$$
#
#Densidade de transição dos estados:
#
#$$p(\Sigma_t^{-1}|\Sigma_{t-1}^{-1},v,d,C)\sim W(v,S_{t-1})$$
#
#sendo que:
#
#$$S_{t-1}=\frac{1}{v}(C^{1/2})(\Sigma_{t-1})^{d}(C^{1/2})^{\prime}$$
#

# Parameters
const T,N = size(yt)
#d     = par[1]^2/(1+par[1]^2)        # global persistence parameter;
#v     = K+abs(par[2])                # degrees of freedom;
d = par[1]
v = par[2]
const Neis = 100
const iv   = 1/v
const LOG2PI = 1.83787706640934533908193770912475883960723876953125;
const LOG2   = 0.69314718055994528622676398299518041312694549560546875;
const LOGPI  = 1.1447298858494001638774761886452324688434600830078125;
const Arg    = v-K+1:v
const df_p1  = v+1                     # degrees of freedom from Wishart EIS initial sampler. See eq (17)
const inds   = find(tril!(ones(K,K)))  # index from covariance matrix to EIS parameters (auxiliary parameters)
const Nb     = length(inds)            # number of estimated parameters from the covariance matrix or from the \Gamma matrix filled with (k^2+k)/2 auxiliary parameters
const ind_   = find(eye(K))            # index from a diagonal KxK matrix
const itmax  = 3;                      # maximum number of iterations;
const tol    = 0.01;                   # EIS tolerance to stop iterations;

# Parameters and variables for EIS iterations
it      = 0
diff    = 100                         # initial difference;
bet     = zeros(Nb+1,T)               # initialize vector to save EIS parameters;
betas_0 = bet                         # initialize EIS parameters at zero for initial sampler. See paragraph bellow eq. (17);
VINI    = Array{Float64,3}(K,K,Neis)
for i=1:Neis
    VINI[:,:,i]    = Vini             # state at t=0. Initial condition is treated as given
end
Y       = nan(Neis,1)                 # dependent variable for EIS regression;
XX      = NaN(Neis,Nb+1,T)            # set space for Neis draws of EIS parameters at each period t;
Ytm1    = NaN(Neis,1)                 # set space for Neis evaluations of the EIS objective function.
indd    = zeros(Int64,1,K+1)
for i=1:K
    indd(i)=find(inds==ind_(i))       # indicate the linear indice in "inds" which corresponds to a diagonal parameter;
end
indd[end] = Nb+1
cC[inds] = par[3:Nb+2]                # maps parameters to cC matrix in order to recriate matrix cC;
cC       = reshape(cC,(K,K));         # reorganize matrix;
C        = PDMat(cC*cC')

# Begin EIS iterations
while diff>tol && it<itmax
    # Use CRNs
    srand(semente)

    # Initialize EIS sampler. See eq. (17)
    V0     = copy(VINI)                 # initial state is given in this exercise;
    df_eis = [bet[end,:]'+df_p1;df_p1]  # degrees of freedom of EIS samplers from t=1 to t=T. Note that for t=T, df_eis=df_p1, since filter estimate is equal to smoothed estimate at T;

    # Sample states for EIS regressions:
    for t=1:T
        BetMat = zeros(K,K)                             # BetMat will be used as the \Gamma matrix from Eq. (17), which collects (k^2+k)/2 auxiliary parameters from the scale matrix of the EIS sampler
        BetMat[inds] = bet(1:Nb,t);                     # initial sampler starts with zero for EIS parameters;
        BetMat       = BetMat+tril(BetMat,-1)';         # tril(BetMat,-1)' complements the upper triangular part of BetMat with the same elements below the main diagonal of BetMat;
        EISmat       = BetMat+yt(t,:)'*yt(t,:);         # part of the scale matrix of the EIS sampler: \Gamma + yt*yt';
                                                        # this part of the code stays outside the loop for N draws, because EISmat is the same for every replication i;
        df_eis_t    = df_eis(t);
        # Construct EIS moments:
        if t==1 # Exploit the fact that V0 is diagonal
            Vd = V0.^(d/2)
            Veist = Array{Float64,3}(K,K,Neis)
            XXX   = Array{Float64,2}(Neis,Nb+1)
            for  i=1:Neis
                sS  = C.chol[:L]*cV0*(iv/2)
                # Compute symmetric PDMatrix scale matrix
                S   = PDMat(sS*sS')
                EISmatS = EISmat*S
            end
        else
        end
    end
end
