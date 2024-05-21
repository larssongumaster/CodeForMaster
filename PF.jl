using Random

using StochasticAD
using Optimisers
using Distributions
using DistributionsAD
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Zygote
using ForwardDiff
using Measurements
using Plots
using ProgressBars
using Printf

# Euler Maruyama for simulation of x and y #
function EulerMaruyama(theta, N, dt, init, obs_error)
    
    alfa = theta[1]
    beta = theta[2]
    sigma = theta[3]
    
    d = Normal(0,sqrt(dt))

    X = zeros(N)
    X[1] = init

    Y = zeros(N)
    Y[1] = X[1] + rand(obs_error)

    for i in 2:N
        X[i] = X[i - 1] - beta*(X[i - 1] - alfa)*dt + sigma*rand(d)
        Y[i] = X[i] + rand(obs_error)
    end
    return X, Y
end

# stratified resampling (from example) #
function sample_stratified(p, K, sump=1)
    n = length(p)
    U = rand()
    is = zeros(Int, K)
    i = 1
    cw = p[1]
    for k in 1:K
        t = sump * (k - 1 + U) / K
        while cw < t && i < n
            i += 1
            @inbounds cw += p[i]
        end
        is[k] = i
    end
    return is
end

function resample(m, X, W, ω, sample_strategy, use_new_weight=true)
    js = Zygote.ignore(() -> sample_strategy(W, m, ω))
    X_new = X[js]
    if use_new_weight
        # differentiable resampling
        W_chosen = W[js]
        W_new = map(w -> ω * new_weight(w / ω) / m, W_chosen)
    else
        # stop gradient, biased approach
        W_new = fill(ω / m, m)
    end
    X_new
end

function particle_filter(N, Y, theta, dt, obs_error)
   
    X = [rand(Normal(Y[1],1)) for j in 1:N] # particles
    log_w = [log(1/N) for j in 1:N]

    loglik = 0
    T = length(Y) # number of observations

    Xstore = zeros(N,T)

    alfa  = theta[1]
    beta  = theta[2]
    sigma = theta[3]

    d = Normal(0,sqrt(dt))
    prop(x) = x - beta*(x - alfa)*dt + sigma*rand(d)

    for (t, y) in zip(1:T, Y)
        Xstore[:,t] .= X
        # update weights & likelihood using observations
        log_w = [logpdf(Normal(y,obs_error), x) for x in X]
        max_w = maximum(log_w)
        sumw = sum(exp.(log_w .- max_w))
        loglik += max_w + log(sumw) - log(N) # loglikelihood for stability
        
        # update particle states
        if t < T
            X = map(x -> prop(x), X)
        end
    end

    return loglik, Xstore
end

function bootstrap_particle_filter(N, Y, theta, dt, obs_error)
   
    X = [rand(Normal(Y[1],1)) for j in 1:N] # particles
    log_w = [log(1/N) for j in 1:N]

    loglik = 0
    T = length(Y) # number of observations

    Xstore = zeros(N,T)

    alfa  = theta[1]
    beta  = theta[2]
    sigma = theta[3]

    d = Normal(0,sqrt(dt))
    prop(x) = x - beta*(x - alfa)*dt + sigma*rand(d)

    for (t, y) in zip(1:T, Y)
        Xstore[:,t] .= X
        # resample particles
        tmp_X = resample(N,X,exp.(log_w)./sum(exp.(log_w)), 1, sample_stratified)
        
        # update weights & likelihood using observations
        log_w = [logpdf(Normal(y,obs_error), x) for x in X]
        max_w = maximum(log_w)
        sumw = sum(exp.(log_w .- max_w))
        loglik += max_w + log(sumw) - log(N) # loglikelihood for stability
        
        # update particle states
        if t < T
            X = map(x -> prop(x), tmp_X)
            Xstore[:,t+1] .= X
        end
    end

    return loglik, Xstore
end

Random.seed!(123);
theta = [15, 0.01, 0.1];

X, Y = EulerMaruyama(theta, 1000, 1, 0, Normal(0,1));
plot(X)
plot(Y)

# comparison #
theta1 = [12, 0.03, 0.1];
theta2 = [16, 0.02, 0.1];

loglik, X1 = particle_filter(50, Y, theta, 1, 1);
loglik, X2 = bootstrap_particle_filter(50, Y, theta, 1, 1);

p = plot(Y)
for i in 1:50
    plot!(p,X2[i,:])
end    
display(p)

for i in 1:5
    p = plot(Y)
    plot!(p,X2[i,:])
    display(p)
end    
