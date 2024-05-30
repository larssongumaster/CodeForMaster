using Plots

# plot 1 #
x = range(-3, 3, length=100);
y(x) = x.^3 .- 5*x;
yprim(x) = 3*x.^2 .- 5;
tangent(x_0,size) = [y(x_0) + yprim(x_0)*(-size), y(x_0) + yprim(x_0)*(size)]


plot(x,y(x), label="", lw=2, xticks = ([-2*sqrt(5/3), -2, 0, sqrt(5/3)], ["-2√(5/3)", string(-2), string(0),"√(5/3)"]))
vline!([sqrt(5/3)], lw=1, ls=:dash, color="black", label="")
vline!([-2*sqrt(5/3)], lw=1, ls=:dash, color="black", label="")
hline!([-10*sqrt(5/3)/3], lw=1, ls=:dash, color="black", label="")
plot!([-0.2, 0.2],tangent(0,0.2),lw=1, color="red", label="",arrow=true)
plot!([-1.8, -2.2],tangent(-2,-0.2),lw=1, color="red", label="",arrow=true)

savefig("GD1.png")

# plot 2 #
x = range(-1, 2, length=100);
y(x) = x.^4 .- 2*x.^3 .+ 2;

plot(x,y(x), label="", lw=2, xticks = ([0, 1.5], [string(0), string(1.5)]))
vline!([0], lw=1, ls=:dash, color="black", label="")
vline!([1.5], lw=1, ls=:dash, color="black", label="")

savefig("GD2.png")