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
function EulerMaruyama(theta, subjects, N, dt, init, obs_error)
    X  = zeros(subjects, N)
    Y  = zeros(subjects, N)
    tv = zeros(subjects, 3)

    d = Normal(0,sqrt(dt))

    for j in 1:subjects
        alfa  = rand(theta[1])
        beta  = rand(theta[2])
        sigma = rand(theta[3])
        
        tv[j,1] = alfa
        tv[j,2] = beta
        tv[j,3] = sigma 

        X[j,1] = init
        Y[j,1] = X[j,1] + rand(obs_error)

        for i in 2:N
            X[j,i] = X[j,i - 1] - beta*(X[j,i - 1] - alfa)*dt + sigma*rand(d)
            Y[j,i] = X[j,i] + rand(obs_error)
        end
    end
    return X, Y, tv
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

function getStart(Y, subjects, particles, dt, error, prior, chains, ratio);
    # proposed starts from prior #
    prior1 = product_distribution(prior[1], prior[2], prior[3]);
    proposed_starts = rand(prior1, chains*ratio)
    likelihoods = zeros(chains*ratio)

    for i in 1:(chains*ratio)
        for j in 1:subjects
            likelihoods[i] = likelihoods[i] + particle_filter(particles,Y[j,:],proposed_starts[:,i],dt,error)
        end
    end

    idxs = partialsortperm(-1*likelihoods, 1:chains)

    @printf("Proposed values found!\n");

    r = zeros(3,chains)

    for i in 1:chains
        r[:,i] .= proposed_starts[:,idxs[i]]
    end
    
    return r
end

# methods for MH #
function conjugacy(prior, knownVar, data);
    posterior_var = 1 / (1/var(prior) + length(data)/knownVar)
    posterior_mean = posterior_var * (mean(prior)/var(prior) + sum(data)/knownVar)

    return Normal(posterior_mean, sqrt(posterior_var))
end

function pilotGibbs(Y, dt, error, chains, start, particles, iterations, priors, vars, vars_d, log_scale=false);
    subjects = length(Y[:,1])
    likelihood = zeros(chains, iterations)
    time = zeros(chains)

    # individual level #
    individual = zeros(subjects, chains, iterations)

    # population level #
    population = zeros(chains,iterations)

    # common parameters #
    common = zeros(2,chains,iterations)
    for c in 1:chains
        individual[:,c,1] .= start[1,c]
        population[c,1] = start[1,c]
        common[:,c,1] = start[2:3,c]
    end


    for c in 1:chains

        t = time_ns()
        # for conjugacy
        con_prior = Normal(mean(priors[1]), sqrt(var(priors[1])))

        
        # for better performance 
        last_accepted_likelihood = zeros(subjects)

        for i in 1:subjects
            last_accepted_likelihood[i] = particle_filter(particles,Y[i,:],start[:,c],dt,error,log_scale)
        end

        likelihood[c,1] = sum([particle_filter(particles, Y[j,:], [individual[j,c,1], common[1,c,1], common[2,c,1]], dt, error, log_scale) for j in 1:subjects])

        iter = ProgressBar(2:iterations)
        for i in iter
            
            # individual level #
            for j in 1:subjects
                prop_alpha = rand(Normal(individual[j,c,i-1], vars[1]))
                prop_alpha_likelihood = particle_filter(particles, Y[j,:], [prop_alpha, common[1,c,i-1], common[2,c,i-1]], dt, error, log_scale)
                
                a = min(0, (prop_alpha_likelihood       + logpdf(Normal(population[c,i-1], vars[1]), prop_alpha)) 
                        -  (last_accepted_likelihood[j] + logpdf(Normal(population[c,i-1], vars[1]), individual[j,c,i-1])))

                if log(rand(Uniform(0,1))) < a
                    individual[j,c,i] = prop_alpha
                    last_accepted_likelihood[j] = prop_alpha_likelihood
                else
                    individual[j,c,i] = individual[j,c,i-1]
                end

            end

            # common parameters #
            prop_common = [rand(Normal(common[1,c,i-1], vars[2])), rand(Normal(common[2,c,i-1], vars[3]))]

            prop_common_likelihood = logpdf(priors[2],prop_common[1]) + logpdf(priors[3],prop_common[2])
            last_accepted_common_likelihood = logpdf(priors[2],common[1,c,i-1]) + logpdf(priors[3],common[2,c,i-1])

            for j in 1:subjects
                prop_common_likelihood = prop_common_likelihood                   + particle_filter(particles, Y[j, :], [individual[j,c,i], prop_common[1], prop_common[2]]  , dt, error, log_scale)
                last_accepted_common_likelihood = last_accepted_common_likelihood + particle_filter(particles, Y[j, :], [individual[j,c,i], common[1,c,i-1], common[2,c,i-1]], dt, error, log_scale)
            end

            a = min(0, prop_common_likelihood - last_accepted_common_likelihood)

            if log(rand(Uniform(0,1))) < a
                common[:,c,i] = prop_common
            else
                common[:,c,i] = common[:,c,i-1]
            end

            # population level #
            posterior = conjugacy(con_prior, vars_d[1], individual[:,c,i]);
            population[c,i] = rand(posterior)
            con_prior = posterior

            likelihood[c,i] = sum([particle_filter(particles, Y[j,:], [individual[j,c,i], common[1,c,i], common[2,c,i]], dt, error, log_scale) for j in 1:subjects])
            set_description(iter, string(@sprintf("Chain: %d / %d", c, chains)))
        end
        time[c] = (time_ns() - t)/1.0e9 
    end

    return individual, population, common, likelihood, time
end

# methods for AD #
function pilotAD(Y, dt, error, chains, start, particles, iterations, log_scale = false);

    subjects = length(Y[:,1])
    result = zeros(subjects+3, chains, iterations)    
    likelihood = zeros(chains, iterations)
    time = zeros(chains)

    fun(p) = -1*sum([particle_filter(particles, Y[i,:], [p[i], p[subjects+2], p[subjects+3]], dt, error, log_scale) for i in 1:subjects])

    for i in 1:chains
        t = time_ns()

        start_long = zeros(subjects+3)
        for j in 1:subjects
            start_long[j] = start[1,i]
        end
        start_long[subjects+1] = start[1,i]
        start_long[subjects+2] = start[2,i]
        start_long[subjects+3] = start[3,i]
    

        if log_scale    
            m = StochasticModel(p -> fun(p), log.(start_long))
        else
            m = StochasticModel(p -> fun(p), start_long)
        end
        o = Adam() # use Adam for optimization
        s = Optimisers.setup(o, m)
        iter = ProgressBar(1:iterations)
        for j in iter
            # Perform a gradient step
            Optimisers.update!(s, m, stochastic_gradient(m))
            result[:, i, j] = m.p # Our optimized value of p
            result[subjects+1,i,j] = mean(result[1:subjects,i,j])
            likelihood[i,j] = -1*fun(m.p)
            set_description(iter, string(@sprintf("Chain: %d / %d", i, chains)))
        end
        time[i] = (time_ns() - t)/1.0e9
    end

    return result, likelihood, time
end

# comparison #
function pilotComp(path, true_val, Y, subjects, dt, error, chains, ratio, particles, iterations, prior, vars, vars_d);
    # writing to file and plotting data #
    f = open(path*"/file.txt", "w");
    write(f, @sprintf("True Vals: [%f,%f,%f]\n", true_val[1], true_val[2], true_val[3]));
    write(f, @sprintf("Subjects: %d, Chains: %d, Particles: %d, Iterations: %d\n\n", subjects, chains, particles, iterations));
    
    p = plot(Y[1,:], title="", label="", xlabel="Time", xticks = (Int.(LinRange(0, length(Y[1,:]), 5)), string.(Int.(LinRange(0, length(Y[1,:])*dt, 5)))), ylabel="Process value", lw = 2, guidefontsize=15)
    for i in 2:subjects
         plot!(p,Y[i,:],label="", lw=2);
    end
    display(p)
    savefig(path*"/data.png")
    
    # start values for chains #
    start = getStart(Y, subjects, particles, dt, error, prior, chains, ratio);

    # mh #
    ind, pop, com, lMH, tMH = pilotGibbs(Y, dt, error, chains, start, particles, iterations, prior, vars, vars_d, false);
    @printf("Time Elapsed for MH: %.0f ± %.0f s\n", mean(tMH), std(tMH))
    write(f, @sprintf("Time Elapsed for MH: %.0f ± %.0f s\n", mean(tMH), std(tMH)))

    plot_path(pop,"alpha",true,true_val[1])
    savefig(path*"/MH_alpha.png")
    plot_path(com[1,:,:],"beta",true,true_val[2])
    savefig(path*"/MH_beta.png")
    plot_path(com[2,:,:],"sigma",true,true_val[3])
    savefig(path*"/MH_sigma.png")

    plot_path(lMH,"loglikelihood",true)
    savefig(path*"/MH_likelihood.png")

    burnin = Int(round(iterations/2))
    resMH = [mean(pop[:,burnin:iterations]), mean(com[1,:,burnin:iterations]), mean(com[2,:,burnin:iterations])]
    varMH = [ std(pop[:,burnin:iterations]),  std(com[1,:,burnin:iterations]),  std(com[2,:,burnin:iterations])]

    # ad1 #
    rAD, lAD, tAD = pilotAD(Y, dt, error, chains, start, particles, iterations, true);
    @printf("Time Elapsed for SAD1: %.0f ± %.0f s\n", mean(tAD), std(tAD))
    write(f, @sprintf("Time Elapsed for SAD1: %.0f ± %.0f s\n", mean(tAD), std(tAD)))
    
    plot_path(exp.(rAD[subjects + 1,:,:]),"alpha",false,true_val[1])
    savefig(path*"/AD_alpha.png")
    plot_path(exp.(rAD[subjects + 2,:,:]),"beta",false,true_val[2])
    savefig(path*"/AD_beta.png")
    plot_path(exp.(rAD[subjects + 3,:,:]),"sigma",false,true_val[3])
    savefig(path*"/AD_sigma.png")
    
    plot_path(lAD,"loglikelihood",false)
    savefig(path*"/AD_likelihood.png")
    
    resAD1 = [mean(exp.(rAD[subjects + 1,:,iterations])), mean(exp.(rAD[subjects + 2,:,iterations])), mean(exp.(rAD[subjects + 3,:,iterations]))]    
    varAD1 = [ std(exp.(rAD[subjects + 1,:,iterations])),  std(exp.(rAD[subjects + 2,:,iterations])),  std(exp.(rAD[subjects + 3,:,iterations]))]    

    # ad2 #
    iterations3 = Int(round(iterations/(mean(tAD)/mean(tMH))))
    elapsed3  = (mean(tAD/(iterations/iterations3)))
    elapsed3v =  (std(tAD/(iterations/iterations3)))

    @printf("Time Elapsed for AD2: %.0f ± %.0f s\n", elapsed3, elapsed3v)    
    write(f, @sprintf("Time Elapsed for AD1: %.0f ± %.0f s\n", elapsed3, elapsed3v))

    # plot comparison #
    plot_comp(pop, exp.(rAD[subjects + 1,:,:]),iterations3,"alpha",true_val[1])
    savefig(path*"/comp_alpha.png")
    plot_comp(com[1,:,:], exp.(rAD[subjects + 2,:,:]),iterations3,"beta",true_val[2])
    savefig(path*"/comp_beta.png")
    plot_comp(com[2,:,:], exp.(rAD[subjects + 3,:,:]),iterations3,"sigma",true_val[3])
    savefig(path*"/comp_sigma.png")

    plot_comp(lMH,lAD,iterations3,"loglikelihood")
    savefig(path*"/comp_likelihood.png")

    resAD2 = [mean(exp.(rAD[subjects + 1,:,iterations3])),mean(exp.(rAD[subjects + 2,:,iterations3])), mean(exp.(rAD[subjects + 3,:,iterations3]))]
    varAD2 = [ std(exp.(rAD[subjects + 1,:,iterations3])), std(exp.(rAD[subjects + 2,:,iterations3])),  std(exp.(rAD[subjects + 3,:,iterations3]))]
    
    absMH  = abs.(true_val-resMH)
    absAD1 = abs.(true_val-resAD1)
    absAD2 = abs.(true_val-resAD2)

    meanLikMH  = mean(lMH[burnin:iterations])
    meanLikAD1 = mean(lAD[:,iterations])
    meanLikAD2 = mean(lAD[:,iterations3])

    varLikMH  = std(lMH[burnin:iterations])
    varLikAD1 = std(lAD[:,iterations])
    varLikAD2 = std(lAD[:,iterations3])

    write(f, @sprintf("Component Results for MH: %.2f, %.4f, %.4f\n", resMH[1], resMH[2], resMH[3]))
    write(f, @sprintf("Component Results for AD1: %.2f, %.4f, %.4f\n", resAD1[1], resAD1[2], resAD1[3]))
    write(f, @sprintf("Component Results for AD2: %.2f, %.4f, %.4f\n", resAD2[1], resAD2[2], resAD2[3]))

    write(f, @sprintf("Component Absolute value from True val for MH: %.2f, %.4f, %.4f\n", absMH[1], absMH[2], absMH[3]))
    write(f, @sprintf("Component Absolute value from True val for AD1: %.2f, %.4f, %.4f\n", absAD1[1], absAD1[2], absAD1[3]))
    write(f, @sprintf("Component Absolute value from True val for AD2: %.2f, %.4f, %.4f\n", absAD2[1], absAD2[2], absAD2[3]))

    write(f, @sprintf("loglikelihood at Component Results for MH: %f\n", meanLikMH))
    write(f, @sprintf("loglikelihood at Component Results for AD1: %f\n", meanLikAD1))
    write(f, @sprintf("loglikelihood at Component Results for AD2: %f\n", meanLikAD2))
    close(f)

    f = open(path*"/tab.txt", "w");
    write(f, @sprintf("Gibbs & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %f ± %f \\\\ \n", mean(tMH), std(tMH), resMH[1], varMH[1], resMH[2], varMH[2], resMH[3], varMH[3], meanLikMH,varLikMH))
    write(f, @sprintf("SAD1 & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %f ± %f \\\\ \n", mean(tAD), std(tAD), resAD1[1], varAD1[1], resAD1[2], varAD1[2], resAD1[3], varAD1[3], meanLikAD1,varLikAD1))
    write(f, @sprintf("SAD2 & %.0f & %.0f s & (%.2f ± %.2f, %.4f ± %.4f, %.4f ± %.4f)  & %f ± %f \\\\ [1ex]\n", elapsed3, elapsed3v, resAD2[1], varAD2[1], resAD2[2], varAD2[2], resAD2[3], varAD2[3], meanLikAD2,varLikAD2))
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
    
    pl = plot(avMH[:], xlabel="Iterations", ylabel=title, label="Gibbs", lw = 2, lc=:blue, guidefontsize=15)
    plot!(avAD[:], label="SAD", lw = 2, lc=:red)
    vline!([comp],label = "Cut for Comparison", lw = 1.5, lc=:black, linestyle=:dash)
    if val != -1000
        hline!([val],label = "True val", lw = 1.5, lc=:black)
    end
    display(pl)
end


# run 1 #
theta   = [15, 0.01, 0.1];
theta_v = [5, 0, 0];
theta_d = [Normal(theta[i],theta_v[i]) for i in 1:3];

subjects = 10;
obs = 20;
dt = 50;
start = 0;
error = 1;

chains = 5;
ratio = 3;
particles = 200;
iterations = 3000;

vars = [2,0.0005,0.01];

prior_alpha = Normal(10,10);
prior_beta  = Uniform(0,0.1);
prior_sigma = Uniform(0,0.5);

prior = [prior_alpha, prior_beta, prior_sigma];

Random.seed!(123);
X, Y, true_val = EulerMaruyama(theta_d, subjects, obs, dt, start, Normal(0,error));    

path = "chained/mixed11";
r, a, l = pilotComp(path, theta, Y, subjects, dt, error, chains, ratio, particles, iterations, prior, vars, theta_v[1]);

# run 2 (more variance on population parameter) #
theta   = [15, 0.01, 0.1];
theta_v = [10, 0, 0];
theta_d = [Normal(theta[i],theta_v[i]) for i in 1:3];

subjects = 10;
obs = 20;
dt = 50;
start = 0;
error = 1;

chains = 5;
ratio = 3;
particles = 200;
iterations = 3000;

vars = [2,0.0005,0.01];

prior_alpha = Normal(10,10);
prior_beta  = Uniform(0,0.1);
prior_sigma = Uniform(0,0.5);

prior = [prior_alpha, prior_beta, prior_sigma];

Random.seed!(123);
X, Y, true_val = EulerMaruyama(theta_d, subjects, obs, dt, start, Normal(0,error));

path = "chained/mixed12";
r, a, l = pilotComp(path, theta, Y, subjects, dt, error, chains, ratio, particles, iterations, prior, vars, theta_v[1]);

# run 4 (less variance on population parameter) #
theta   = [15, 0.01, 0.1];
theta_v = [1, 0, 0];
theta_d = [Normal(theta[i],theta_v[i]) for i in 1:3];

subjects = 10;
obs = 20;
dt = 50;
start = 0;
error = 1;

chains = 5;
ratio = 3;
particles = 200;
iterations = 3000;

vars = [2,0.0005,0.01];

prior_alpha = Normal(10,10);
prior_beta  = Uniform(0,0.1);
prior_sigma = Uniform(0,0.5);

prior = [prior_alpha, prior_beta, prior_sigma];

Random.seed!(123);
X, Y, true_val = EulerMaruyama(theta_d, subjects, obs, dt, start, Normal(0,error));

path = "chained/mixed14";
r, a, l = pilotComp(path, theta, Y, subjects, dt, error, chains, ratio, particles, iterations, prior, vars, theta_v[1]);

# run 5 (more iterations) #
theta   = [15, 0.01, 0.1];
theta_v = [5, 0, 0];
theta_d = [Normal(theta[i],theta_v[i]) for i in 1:3];

subjects = 10;
obs = 20;
dt = 50;
start = 0;
error = 1;

chains = 5;
ratio = 3;
particles = 200;
iterations = 10000;

vars = [2,0.0005,0.01];

prior_alpha = Normal(20,10);
prior_beta  = Uniform(0,0.1);
prior_sigma = Uniform(0,0.5);

prior = [prior_alpha, prior_beta, prior_sigma];

Random.seed!(123);
X, Y, true_val = EulerMaruyama(theta_d, subjects, obs, dt, start, Normal(0,error));    

path = "chained/mixed15";
r, a, l = pilotComp(path, theta, Y, subjects, dt, error, chains, ratio, particles, iterations, prior, vars, theta_v[1]);
