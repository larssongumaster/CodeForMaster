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

function particle_filter(N, Y, theta, dt, obs_error, log_scale = false)
   
    X = [rand(Normal(Y[1],1)) for j in 1:N] # particles
    log_w = [log(1/N) for j in 1:N]

    T = length(Y)

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

# functions for pilot studies #
function getStart(Y, particles, dt, error, prior, chains, ratio);
    # proposed starts from prior #
    proposed_starts = rand(prior, chains*ratio)
    likelihoods = zeros(chains*ratio)

    for i in 1:(chains*ratio)
        likelihoods[i] = particle_filter(particles,Y,proposed_starts[:,i],dt,error)
    end

    idxs = partialsortperm(-1*likelihoods, 1:chains)

    @printf("Proposed values found!\n");

    r = zeros(3,chains)

    for i in 1:chains
        r[:,i] .= proposed_starts[:,idxs[i]]
    end
    
    return r
end

function pilotMH(Y, dt, error, chains, start, particles, cut, iterations, prior, vars, log_scale=false);
    
    result = zeros(length(rand(prior)), chains, iterations)    
    likelihood = zeros(chains, iterations)
    time = zeros(chains)

    loglik = 0;

    # for adaptive #
    epsilon = 10^-8
    sd = ((2.4)^2)/3;

    # actual algorithm
    for i in 1:chains
        t = time_ns()   
        result[:,i,1] .= start[:,i]
        likelihood[i,1] = particle_filter(particles,Y,result[:,i,1],dt,error,log_scale)

        iter = ProgressBar(2:iterations)
        loglik_old = likelihood[i,1];
 
        # check not infinite with error message 
        #if loglik_old > BIG
        #    return
        #end

        for j in iter

            if j > cut
                vars = [cov(result[1, i, 1:j], result[1, i, 1:j]) cov(result[1, i, 1:j], result[2, i, 1:j]) cov(result[1, i, 1:j], result[3, i, 1:j]);
                        cov(result[2, i, 1:j], result[1, i, 1:j]) cov(result[2, i, 1:j], result[2, i, 1:j]) cov(result[2, i, 1:j], result[3, i, 1:j]);
                        cov(result[3, i, 1:j], result[1, i, 1:j]) cov(result[3, i, 1:j], result[2, i, 1:j]) cov(result[3, i, 1:j], result[3, i, 1:j])]
                vars = sd * (vars + [epsilon 0 0; 0 epsilon 0; 0 0 epsilon])
            end

            prop_args = rand(MvNormal(result[:,i,j-1], vars))
            loglik = particle_filter(particles, Y, prop_args, dt, error);

            acceptance = min(0, loglik     + logpdf(prior,prop_args) 
                              - loglik_old - logpdf(prior,result[:,i,j-1]))
                
            if log(rand(Uniform(0,1))) < acceptance 
                result[:,i,j] = prop_args
                loglik_old = loglik;
            else
                result[:,i,j] = result[:,i,j-1]
            end

            likelihood[i,j] = particle_filter(particles,Y,result[:,i,j],dt,error,log_scale)

            set_description(iter, string(@sprintf("Chain: %d / %d", i, chains)))
        end
        time[i] = (time_ns() - t)/1.0e9
    end
    return result, likelihood, time;
end

function pilotAD(Y, dt, error, chains, start, particles, iterations, prior, log_scale = false);
    fun(p) = -1*particle_filter(particles, Y, p, dt, error, log_scale)

    result = zeros(length(rand(prior)), chains, iterations)    
    likelihood = zeros(chains, iterations)
    time = zeros(chains)

    for i in 1:chains
        t = time_ns()
        if log_scale    
            m = StochasticModel(p -> fun(p), log.(start[:,i]))
        else
            m = StochasticModel(p -> fun(p), start[:,i])
        end
        o = Adam() # use Adam for optimization
        s = Optimisers.setup(o, m)
        iter = ProgressBar(1:iterations)
        for j in iter
            # Perform a gradient step
            Optimisers.update!(s, m, stochastic_gradient(m))
            result[:, i, j] = m.p # Our optimized value of p
            likelihood[i,j] = particle_filter(particles, Y, m.p, dt, error, log_scale)
            set_description(iter, string(@sprintf("Chain: %d / %d", i, chains)))
        end
        time[i] = (time_ns() - t)/1.0e9
    end

    return result, likelihood, time

end

function pilotComp(path, true_val, Y, dt, error, chains, ratio, particles, iterations, prior, vars, cut)

    # writing to file and plotting data#
    f = open(path*"/file.txt", "w");
    write(f, @sprintf("True Vals: [%f,%f,%f]\n", true_val[1], true_val[2], true_val[3]));
    write(f, @sprintf("Chains: %d, Particles: %d, Iterations: %d\n\n", chains, particles, iterations));
    
    p = plot(Y, title="", label="", xlabel="Time", xticks = (Int.(LinRange(0, length(Y), 5)), string.(Int.(LinRange(0, length(Y)*dt, 5)))), ylabel="Process value", lw = 2, guidefontsize=15)
    display(p)
    savefig(path*"/data.png")

    # start values for chains #
    start = getStart(Y, particles, dt, error, prior, chains, ratio);

    # mh #
    rMH, lMH, tMH = pilotMH(Y, dt, error, chains, start, particles, cut, iterations, prior, vars);
    @printf("Time Elapsed for MH: %.0f ± %.0f s\n", mean(tMH), std(tMH))
    write(f, @sprintf("Time Elapsed for MH: %.0f ± %.0f s\n", mean(tMH), std(tMH)))

    plot_path(rMH[1,:,:],"alpha",true,true_val[1])
    savefig(path*"/MH_alpha.png")
    plot_path(rMH[2,:,:],"beta",true,true_val[2])
    savefig(path*"/MH_beta.png")
    plot_path(rMH[3,:,:],"sigma",true,true_val[3])
    savefig(path*"/MH_sigma.png")

    plot_path(lMH,"loglikelihood",true)
    savefig(path*"/MH_likelihood.png")

    burnin = Int(round(iterations/2))
    resMH = [mean(rMH[1,:,burnin:iterations]),mean(rMH[2,:,burnin:iterations]),mean(rMH[3,:,burnin:iterations])]
    varMH = [ std(rMH[1,:,burnin:iterations]), std(rMH[2,:,burnin:iterations]), std(rMH[3,:,burnin:iterations])]

    # ad1 #
    rAD, lAD, tAD = pilotAD(Y, dt, error, chains, start, particles, iterations, prior, true);
    @printf("Time Elapsed for AD1: %.0f ± %.0f s\n", mean(tAD), std(tAD))    
    write(f, @sprintf("Time Elapsed for AD1: %.0f ± %.0f s\n", mean(tAD), std(tAD)))

    plot_path(exp.(rAD[1,:,:]),"alpha",false,true_val[1])
    savefig(path*"/AD_alpha.png")
    plot_path(exp.(rAD[2,:,:]),"beta",false,true_val[2])
    savefig(path*"/AD_beta.png")
    plot_path(exp.(rAD[3,:,:]),"sigma",false,true_val[3])
    savefig(path*"/AD_sigma.png")

    plot_path(lAD,"loglikelihood",false)
    savefig(path*"/AD_likelihood.png")

    resAD1 = [mean(exp.(rAD[1,:,iterations])),mean(exp.(rAD[2,:,iterations])),mean(exp.(rAD[3,:,iterations]))]
    varAD1 = [ std(exp.(rAD[1,:,iterations])), std(exp.(rAD[2,:,iterations])), std(exp.(rAD[3,:,iterations]))]

    # ad2 #
    iterations3 = Int(round(iterations/(mean(tAD)/mean(tMH))))
    elapsed3  = (mean(tAD/(iterations/iterations3)))
    elapsed3v =  (std(tAD/(iterations/iterations3)))

    @printf("Time Elapsed for AD2: %.0f ± %.0f s\n", elapsed3, elapsed3v)    
    write(f, @sprintf("Time Elapsed for AD1: %.0f ± %.0f s\n", elapsed3, elapsed3v))

    # plot comparison #
    plot_comp(rMH[1,:,:], exp.(rAD[1,:,:]),iterations3,"alpha",true_val[1])
    savefig(path*"/comp_alpha.png")
    plot_comp(rMH[2,:,:], exp.(rAD[2,:,:]),iterations3,"beta",true_val[2])
    savefig(path*"/comp_beta.png")
    plot_comp(rMH[3,:,:], exp.(rAD[3,:,:]),iterations3,"sigma",true_val[3])
    savefig(path*"/comp_sigma.png")
    
    plot_comp(lMH, lAD,iterations3,"loglikelihood")
    savefig(path*"/comp_likelihood.png")
    
    
    resAD2 = [mean(exp.(rAD[1,:,iterations3])),mean(exp.(rAD[2,:,iterations3])),mean(exp.(rAD[3,:,iterations3]))]
    varAD2 = [ std(exp.(rAD[1,:,iterations3])), std(exp.(rAD[2,:,iterations3])), std(exp.(rAD[3,:,iterations3]))]
    
    absMH  = abs.(true_val-resMH)
    absAD1 = abs.(true_val-resAD1)
    absAD2 = abs.(true_val-resAD2)

    likelihoodMH = zeros(chains)
    likelihoodAD1 = zeros(chains)
    likelihoodAD2 = zeros(chains)

    for i in 1:chains
        likelihoodMH[i]  = particle_filter(particles, Y, [mean(rMH[1,i,cut:iterations]),   mean(rMH[2,i,cut:iterations]),   mean(rMH[3,i,burnin:iterations])], dt, error)
        likelihoodAD1[i] = particle_filter(particles, Y, [mean(exp.(rAD[1,i,iterations])), mean(exp.(rAD[2,i,iterations])), mean(exp.(rAD[3,i,iterations]))],  dt, error)
        likelihoodAD2[i] = particle_filter(particles, Y, [mean(exp.(rAD[1,i,iterations3])),mean(exp.(rAD[2,i,iterations3])),mean(exp.(rAD[3,i,iterations3]))], dt, error)
    end


    meanLikMH  = mean(likelihoodMH)
    meanLikAD1 = mean(likelihoodAD1)
    meanLikAD2 = mean(likelihoodAD2)

    varLikMH  = std(likelihoodMH)
    varLikAD1 = std(likelihoodAD1)
    varLikAD2 = std(likelihoodAD2)

    write(f, @sprintf("Component Results for MH: %.2f, %.4f, %.4f\n",  resMH[1],  resMH[2],  resMH[3]))
    write(f, @sprintf("Component Results for AD1: %.2f, %.4f, %.4f\n", resAD1[1], resAD1[2], resAD1[3]))
    write(f, @sprintf("Component Results for AD2: %.2f, %.4f, %.4f\n", resAD2[1], resAD2[2], resAD2[3]))

    write(f, @sprintf("Component Absolute value from True val for MH: %.2f, %.4f, %.4f\n",  absMH[1],  absMH[2],  absMH[3]))
    write(f, @sprintf("Component Absolute value from True val for AD1: %.2f, %.4f, %.4f\n", absAD1[1], absAD1[2], absAD1[3]))
    write(f, @sprintf("Component Absolute value from True val for AD2: %.2f, %.4f, %.4f\n", absAD2[1], absAD2[2], absAD2[3]))

    write(f, @sprintf("loglikelihood at Component Results for MH: %f\n",  meanLikMH))
    write(f, @sprintf("loglikelihood at Component Results for AD1: %f\n", meanLikAD1))
    write(f, @sprintf("loglikelihood at Component Results for AD2: %f\n", meanLikAD2))
    close(f)

    f = open(path*"/tab.txt", "w");
    write(f, @sprintf("AMH & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %.4f ± %.4f \\\\ \n", mean(tMH), std(tMH), resMH[1], varMH[1], resMH[2], varMH[2], resMH[3], varMH[3], meanLikMH,varLikMH))
    write(f, @sprintf("SAD1 & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %.4f ± %.4f \\\\ \n", mean(tAD), std(tAD), resAD1[1], varAD1[1], resAD1[2], varAD1[2], resAD1[3], varAD1[3], meanLikAD1,varLikAD1))
    write(f, @sprintf("SAD2 & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %.4f ± %.4f \\\\ [1ex]\n", elapsed3, elapsed3v, resAD2[1], varAD2[1], resAD2[2], varAD2[2], resAD2[3], varAD2[3], meanLikAD2,varLikAD2))
    close(f)

    return [resMH, resAD1, resAD2], [absMH, absAD1, absAD2], [meanLikMH, meanLikAD1, meanLikAD2]
end

# functions for plotting #
function plot_path(arr, title, blue, val = -1000);
    c = length(arr[:,1])
    av = mean(arr, dims = 1);
    
    pl = plot(arr[1,:], xlabel="Iterations", ylabel=title, label="Individual Paths", lw = 1, lc=:gray, guidefontsize=15)
    for i in 2:c
        plot!(arr[i,:], label = "", lw = 1, lc=:gray)
    end
    if val != -1000
        hline!([val],label = "True val", lw = 1.5, lc=:black)
    end
    if blue == true
        plot!(av[:], lw = 2, label="Mean of Chains", lc=:blue)
    else
        plot!(av[:], lw = 2, label="Mean of Chains", lc=:red)
    end
    display(pl)
end

function plot_comp(arrMH, arrAD, comp, title, val = -1000);
    avMH = mean(arrMH, dims = 1);
    #stdMH = std(arrMH, dims = 1);
    
    avAD = mean(arrAD, dims = 1);
    #stdAD = std(arrAD, dims = 1);
    
    pl = plot(avMH[:], xlabel="Iterations", ylabel=title, label="AMH", lw = 2, lc=:blue, guidefontsize=15)
    plot!(avAD[:], label="SAD", lw = 2, lc=:red)
    vline!([comp],label = "Cut for Comparison", lw = 1.5, lc=:black, linestyle=:dash)
    if val != -1000
        hline!([val],label = "True val", lw = 1.5, lc=:black)
    end
    display(pl)
end


# run 1 #
# simulating data #
theta = [15, 0.01, 0.1];
obs = 40;
dt = 50;
start = 0;
error = 1;

Random.seed!(123);
X, Y = EulerMaruyama(theta, obs, dt, start, Normal(0,error));

# comparison #
chains = 25;
ratio = 1;
particles = 200;
iterations = 3000;

prior_alpha = Uniform(0,25);
prior_beta  = Uniform(0,0.05);
prior_sigma = Uniform(0,1);

prior = product_distribution(prior_alpha, prior_beta, prior_sigma);

cut = 300;
vars = [2,0.0005,0.01];

path = "simple11";
r, a, l = pilotComp(path,theta,Y,dt,error,chains,ratio,particles,iterations,prior,vars,cut);

# run 2 #
# simulating data #
theta = [10, 0.005, 0.1];
obs = 40;
dt = 50;
start = 0;
error = 1;

Random.seed!(123);
X, Y = EulerMaruyama(theta, obs, dt, start, Normal(0,error));

# comparison #
chains = 25;
ratio = 5;
particles = 200;
iterations = 3000;

prior_alpha = Uniform(0,15);
prior_beta  = Uniform(0,0.1);
prior_sigma = Uniform(0,1);

prior = product_distribution(prior_alpha, prior_beta, prior_sigma);

cut = 300;
vars = [2,0.0005,0.01];

path = "simple12";
r, a, l = pilotComp(path,theta,Y,dt,error,chains,ratio,particles,iterations,prior,vars,cut);