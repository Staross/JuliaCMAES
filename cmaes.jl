# This is a minimal implementation of the CMA-ES based on purecmaes.m
# The following are the original comments
#
# (mu/mu_w, lambda)-CMA-ES
# CMA-ES: Evolution Strategy with Covariance Matrix Adaptation for
# nonlinear function minimization. To be used under the terms of the
# GNU General Public License (http://www.gnu.org/copyleft/gpl.html).
# Copyright: Nikolaus Hansen, 2003-09.
# e-mail: hansen[at]lri.fr
#
# This code is an excerpt from cmaes.m and implements the key parts
# of the algorithm. It is intendend to be used for READING and
# UNDERSTANDING the basic flow and all details of the CMA-ES
# *algorithm*. Use the cmaes.m code to run serious simulations: it
# is longer, but offers restarts, far better termination options,
# and supposedly quite useful output.
#
# URL: http://www.lri.fr/~hansen/purecmaes.m
# References: See end of file. Last change: October, 21, 2010

function cmaes(objFun::Function, pinit, sigma; lambda=0,stopeval=0,stopDeltaFitness=1e-12)

#   objFun(x) = sum( (x-linspace(0,100,length(x))).^2 )
#   objFun(x) = sum( 0.1*(x[1]-1).^2 + (x[2]-2).^2 )
#    N = 3                # number of objective variables/problem dimension
#    xmean = rand(N)      # objective variables initial point
#    sigma = 0.5          # coordinate wise standard deviation (step size)

    display = 1

    # Input checks
    if( max(size([pinit])) == length([pinit]) )
        pinit = vec([pinit])
    else
       error("pinit should be a vector")
    end

    N = length(pinit)

    if( max(size([sigma])) == length([sigma]) )
       sigma = vec([sigma])
    else
       error("sigma should be a vector")
    end

    if(length(sigma)==1)
        sigma = sigma[1]*ones(N)
    end

    if(size(sigma) != size(pinit))
        error("sigma and pinit have different sizes!")
    end

    initialValue = objFun(pinit)
    if(! (typeof(initialValue) <: FloatingPoint) )
        error("objFun should return a scalar Float")
    end

    xmean = pinit

    stopfitness = 1e-10  # stop if fitness < stopfitness (minimization)

    if(stopeval==0) stopeval = 1e3*N^2 end # stop after stopeval number of function evaluations

    #########
    # Strategy parameter setting: Selection
    if(lambda==0)
        lambda = convert(Integer,4+floor(3*log(N)))  # population size, offspring number
    end
    mu = lambda/2                   # number of parents/points for recombination
    weights = log(mu+1/2)-log(1:mu) # muXone array for weighted recombination
    mu = convert(Integer,floor(mu))
    weights = weights/sum(weights)     # normalize recombination weights array
    mueff=sum(weights)^2/sum(weights.^2) # variance-effectiveness of sum w_i x_i

    #########
    # Strategy parameter setting: Adaptation
    cc = (4 + mueff/N) / (N+4 + 2*mueff/N) # time constant for cumulation for C
    cs = (mueff+2) / (N+mueff+5)  # t-const for cumulation for sigma control
    c1 = 2 / ((N+1.3)^2+mueff)    # learning rate for rank-one update of C
    cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff))  # and for rank-mu update
    damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs # damping for sigma, usually close to 1

    #########
    # Initialize dynamic (internal) strategy parameters and constants

    pc = zeros(N); ps = zeros(N)   # evolution paths for C and sigma

    diagD = sigma/max(sigma);      # diagonal matrix D defines the scaling
    diagC = diagD.^2;

    sigma = max(sigma)

    B = eye(N,N)                        # B defines the coordinate system

    BD = B.*repmat(diagD',N,1)          # B*D for speed up only
    C = diagm(diagC);                   # covariance matrix == BD*(BD)'

    eigeneval = 0                      # track update of B and D
    chiN=N^0.5*(1-1/(4*N)+1/(21*N^2))  # expectation of  ||N(0,I)|| == norm(randn(N,1))

    #init a few things
    arx = zeros(N,lambda)
    arz = zeros(N,lambda)
    arfitness = zeros(lambda)
    arindex = zeros(lambda)

    previousFitness = 0

    iter = 0

    @printf("%i-%i CMA-ES\n",lambda,mu);

    # -------------------- Generation Loop --------------------------------
    counteval = 0  # the next 40 lines contain the 20 lines of interesting code
    while counteval < stopeval

    iter = iter+1

    # Generate and evaluate lambda offspring
    for k=1:lambda
        #      arx[:,k] = xmean + sigma * B * (D .* randn(N,1)) # m + sig * Normal(0,C)

        arz[:,k] = randn(N,1); # resample
        arx[:,k] = xmean + sigma * (BD * arz[:,k]);

        arfitness[k] = objFun( arx[:,k] ) # objective function call
        counteval = counteval+1
    end

    # Sort by fitness and compute weighted mean into xmean
    arindex = sortperm(arfitness)
    arfitness = arfitness[arindex]  # minimization

    # Calculate new xmean, this is selection and recombination
    xold = xmean; # for speed up of Eq. (2) and (3)
    xmean = arx[:,arindex[1:mu]]*weights
    zmean = arz[:,arindex[1:mu]]*weights #==D^-1*B'*(xmean-xold)/sigma

    # Cumulation: Update evolution paths
    ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (B*zmean)          # Eq. (4)
    hsig = norm(ps)/sqrt(1-(1-cs)^(2*counteval))/chiN < 1.4 + 2/(N+1)
    pc = (1-cc)*  pc + hsig* (sqrt(cc*(2-cc)*mueff)/sigma) * (xmean-xold)

    # Adapt covariance matrix C
    artmp = (1/sigma) * (arx[:,arindex[1:mu]] - repmat(xold,1,mu))  # mu difference vectors

      C =( (1-c1-cmu+(1-hsig)*c1*cc*(2-cc)) * C     # regard old matrix
          + c1 * pc*pc'                             # plus rank one update
          + cmu                                     # plus rank mu update
          * sigma^-2 * (arx[:,arindex[1:mu]]-repmat(xold,1,mu))
          * (repmat(weights,1,N) .* (arx[:,arindex[1:mu]]-repmat(xold,1,mu))')  )


    # Adapt step size sigma
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1))

    # Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  # to achieve O(N^2)
        eigeneval = counteval
        C = triu(C) + triu(C,1)' # enforce symmetry
        (D,B) = eig(C)           # eigen decomposition, B==normalized eigenvectors

        diagC = diag(C);
        diagD = sqrt(diagD); # D contains standard deviations now
        BD = B.*repmat(diagD',N,1); # O(n^2)
    end

    #Stop conditions:
    # Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable
    if arfitness[1] <= stopfitness || max(D) > 1e7 * min(D)
      break
    end

    #break if fitness doesn't change much
    if(iter > 1)
        if( abs(arfitness[1]-previousFitness) < stopDeltaFitness)
            break
        end
    end
    previousFitness = arfitness[1]

    #Display some information every 25 iterations
    if(display ==1 && iter % 25 ==0)
        @printf("iter: %d \t fcount: %d \t fval: %2.2e \t axis-ratio: %2.2e \n",iter,counteval, arfitness[1], max(D) / min(D) )
    end

    end # while, end generation loop

    xmin = arx[:, arindex[1]];

    function corcov(C::AbstractMatrix)
        #should check if C is positive semidefinite

        sigma = sqrt(diag(C))
        return C ./ (sigma*sigma')
    end

    if( N<10 )
        print("\nCorrelation matrix:\n")
        println(corcov(C))
    end

    println("\npmin:")
    println(xmin)

#    println("\npmin std ")
#    println( 100*sqrt(diag(C)) ./ xmin )
#    println( sqrt(diag(C))  )

    println("\nfmin:")
    println(arfitness[1])

    return xmin

end

# ---------------------------------------------------------------
### REFERENCES
#
# Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution
# Strategy on Multimodal Test Functions.  Eighth International
# Conference on Parallel Problem Solving from Nature PPSN VIII,
# Proceedings, pp. 282-291, Berlin: Springer.
# (http://www.bionik.tu-berlin.de/user/niko/ppsn2004hansenkern.pdf)
#
# Further references:
# Hansen, N. and A. Ostermeier (2001). Completely Derandomized
# Self-Adaptation in Evolution Strategies. Evolutionary Computation,
# 9(2), pp. 159-195.
# (http://www.bionik.tu-berlin.de/user/niko/cmaartic.pdf).
#
# Hansen, N., S.D. Mueller and P. Koumoutsakos (2003). Reducing the
# Time Complexity of the Derandomized Evolution Strategy with
# Covariance Matrix Adaptation (CMA-ES). Evolutionary Computation,
# 11(1).  (http://mitpress.mit.edu/journals/pdf/evco_11_1_1_0.pdf).
