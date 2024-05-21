using Random
using Distributions
using Plots

Random.seed!(123);

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

theta = [15, 0.01, 0.1];
dt = 1;
X, Y = EulerMaruyama(theta, 2000, dt, 0, Normal(0,1));

plot(Y, title="", label="Observable Process", xlabel="Time", xticks = (Int.(LinRange(0, length(Y), 5)), string.(Int.(LinRange(0, length(Y)*dt, 5)))), ylabel="Process value", lw = 2)
plot!(X, label="Latent Process", lw = 2)
savefig("OU1.png")

X, Y = EulerMaruyama(theta, 1000, 1, 0, Normal(0,1));

plot(Y, title="", label="Observable Process", xlabel="Time", xticks = (Int.(LinRange(0, length(Y), 5)), string.(Int.(LinRange(0, length(Y)*dt, 5)))), ylabel="Process value", lw = 2)
plot!(X, label="Latent Process", lw = 2)
savefig("OU2.png")
