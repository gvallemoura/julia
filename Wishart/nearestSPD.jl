function nearestSPD(A::Symmetric{Float64,Array{Float64,2}})
# nearestSPD - the nearest [in Frobenius norm()] Symmetric Positive Definite matrix to A
# usage: Ahat = nearestSPD[A]
#
# From Higham: "The nearest symmetric positive semidefinite matrix in the
# Frobenius norm to an arbitrary real matrix A is shown to be [B + H]/2
# where H is the symmetric polar factor of B=(A + A')/2."
#
# http://www.sciencedirect.com/science/article/pii/0024379588902236
#
# arguments: (input())
#  A - square matrix, which will be converted to the nearest Symmetric
#    Positive Definite Matrix.
#
# Arguments: (output)
#  Ahat - The matrix chosen as the nearest SPD matrix to A.


# symmetrize A into B
B = convert(Array,A)

# Compute the symmetric polar factor of B. Call it H.
# Clearly H is itself SPD.
U,Sigma,V = svd(B)
H = V*diagm(Sigma)*V'

# get Ahat in the above formula
Ahat = Symmetric((B+H)/2)

# ensure symmetry
# Ahat = (Ahat + Ahat')/2

# test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
k = 1
p = 1
while p>0
    k = k+1
    try
        Ahat = PDMat(Ahat)
        # println("Obaa!!")
        p = -1
    catch
        # Ahat failed the chol test. It must have been just a hair off
        # due to floating point trash, so it is simplest now just to
        # tweak by adding a tiny multiple of an identity matrix.
        valor,Vec = eig(Ahat)
        mineig = minimum(valor)
        Ahat = Ahat + (-mineig*k.^2 + sqrt(eps(mineig)))*eye(size(A,1))

    end
end
return Ahat
end
