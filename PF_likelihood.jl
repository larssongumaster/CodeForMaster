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

function particle_filter(N, Y, theta, dt, obs_error, log_scale=false)
   
    X = [rand(Normal(Y[1],1)) for j in 1:N] # particles
    log_w = [log(1/N) for j in 1:N]

    loglik = 0
    T = length(Y) # number of observations

    if log_scale
        alfa  = exp(theta[1])
        beta  = exp(theta[2])
        sigma = exp(theta[3])
    else 
        alfa  = theta[1]
        beta  = theta[2]
        sigma = theta[3]
    end

    d = Normal(0,sqrt(dt))
    prop(x) = x - beta*(x - alfa)*dt + sigma*rand(d)

    for (t, y) in zip(1:T, Y)
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
        end
    end

    return loglik
end

function likelihood_test(name, Y, theta, dt, obs_error, log_scale=false)

    particles = 1:50
    reps = 1000

    index = 1
    var_result  = zeros(length(particles))
    time_result = zeros(length(particles))
    
    ind_result  = zeros(2,reps)

    # add time 
    for N in ProgressBar(particles)
        for rep in 1:reps
            t = time_ns()
            ind_result[1,rep] = particle_filter(N*20, Y, theta, dt, obs_error)
            ind_result[2,rep] = (time_ns() - t)/1.0e9
        end
        var_result[index]  = var(ind_result[1,:])
        time_result[index] = mean(ind_result[2,:])
        index = index + 1
    end

    p = plot(var_result, label = "", xlabel = "Particles", ylabel = "Variance", xticks = (10:10:50, string.(200:200:1000)), lc=:blue, lw=2, guidefontsize=15)
    display(p)
    savefig(name*"-var.png")

    p = plot(time_result, label = "", xlabel = "Particles", ylabel = "Time in Seconds", xticks = (10:10:50, string.(200:200:1000)), lc=:red, lw=2, guidefontsize=15)
    display(p)
    savefig(name*"-time.png")

    return
end

function gradient_test(name, Y, theta, dt, obs_error, log_scale=false)

    particles = 1:50
    reps = 200

    index = 1
    var_result  = zeros(3,length(particles))
    ind_result  = zeros(3 ,reps)

    for N in ProgressBar(particles)
        fun(p) = particle_filter(20*N, Y, p, dt, obs_error, log_scale)
        for rep in 1:reps
            derivates = derivative_estimate(fun, theta)

            ind_result[1,rep] = derivates[1]
            ind_result[2,rep] = derivates[2]
            ind_result[3,rep] = derivates[3]
        end
        var_result[1,index] = var(ind_result[1,:])
        var_result[2,index] = var(ind_result[2,:])
        var_result[3,index] = var(ind_result[3,:])
        index = index + 1
    end

    if log_scale
        p = plot(var_result[1,:], label = "", xlabel = "Particles", ylabel = "Variance of α", xticks = (10:10:50, string.(200:200:1000)), lc=:red, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-alpha.png")

        p = plot(var_result[2,:], label = "", xlabel = "Particles", ylabel = "Variance of β", xticks = (10:10:50, string.(200:200:1000)), lc=:red, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-beta.png")

        p = plot(var_result[3,:], label = "", xlabel = "Particles", ylabel = "Variance of σ", xticks = (10:10:50, string.(200:200:1000)), lc=:red, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-sigma.png")    
    else
        p = plot(var_result[1,:], label = "", xlabel = "Particles", ylabel = "Variance of α", xticks = (10:10:50, string.(200:200:1000)), lc=:blue, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-alpha.png")

        p = plot(var_result[2,:], label = "", xlabel = "Particles", ylabel = "Variance of β", xticks = (10:10:50, string.(200:200:1000)), lc=:blue, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-beta.png")

        p = plot(var_result[3,:], label = "", xlabel = "Particles", ylabel = "Variance of σ", xticks = (10:10:50, string.(200:200:1000)), lc=:blue, lw=2, guidefontsize=15)
        display(p)
        savefig(name*"-sigma.png")
    end

    return
end


Random.seed!(123);
dt = 10;
theta = [15, 0.01, 0.1];

X, Y = EulerMaruyama(theta, 100, dt, 0, Normal(0,1));
plot(X)
plot(Y)

# other theta values for comparison #
theta1 = [14, 0.02, 0.15];
theta2 = [20, 0.1, 0.1];

likelihood_test("L-15-001-01",  Y, theta,  dt, 1);
likelihood_test("L-14-002-015", Y, theta1, dt, 1);
likelihood_test("L-20-01-01",   Y, theta2, dt, 1);

gradient_test("G-15-001-01",  Y, theta, dt, 1, false);
gradient_test("GL-15-001-01", Y, log.(theta), dt, 1, true);

gradient_test("G-14-002-015",  Y, theta1, dt, 1, false);
gradient_test("GL-14-002-015", Y, log.(theta1), dt, 1, true);

gradient_test("G-20-01-01",  Y, theta2, dt, 1, false);
gradient_test("GL-20-01-01", Y, log.(theta2), dt, 1, true);

