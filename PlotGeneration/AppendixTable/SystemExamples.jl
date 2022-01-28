using CairoMakie
using DifferentialEquations
using Random: MersenneTwister

mrng = MersenneTwister(1234)

##################################
####### Plot Setup ###############
##################################

noto_sans = "./resources/NotoSans-Regular.ttf"

tickfontsize    = 28
labelfontsize   = 32

#f = Figure(font=noto_sans, resolution=(800,400))
f = Figure(resolution = (2000, 400))

################################
########## Morse Function ######
################################

# (x^2 - 1)(x^2 - 4)
# f(x) = (x-1) * (x+1) * (x-2) * (x+2)
g(x) = 4 * x^3 - 10*x

function morseFlow!(du, u, μ, t)
    g(x) = 4.0 * x^3 - 10.0 * x

    du .= -0.5 .* g.(u) 
end

curve_count = 50
u0 = 3 * randn(mrng, curve_count)

solution_curves = zeros(101, curve_count)
timings = 0:0.01:1

for i in 1:curve_count
    tspan = (0.0, 1.0)
    prob = ODEProblem(morseFlow!, [u0[i]], tspan)

    sol = solve(prob, saveat=0.01)
    solution_curves[:,i] = sol[1,:]
end

ax1 = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="τ", ylabel = "x",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            ylims = (-5.0, 5.0))

for i in 1:curve_count
    lines!(ax1, timings, solution_curves[:,i], color=:black)
end
xlims!(ax1,(0,1))


################################
##### Van Der Pol Oscillator ###
################################

function vanderpol!(du,u,μ,t)
    x,y = u
    du[1] = μ * (x - x^3/3 - y)
    du[2] = x/μ
end

curve_count = 20
u0 = 5 * rand(mrng, 2,curve_count) .- 2.5

solution_curves = zeros(201, 2, curve_count)

for i in 1:curve_count
    tspan = (0.0, 10.0)
    prob = ODEProblem(vanderpol!, u0[:,i], tspan, 1)

    sol = solve(prob, saveat=0.05)
    solution_curves[:, 1, i] = sol[1,:]
    solution_curves[:, 2, i] = sol[2,:]
end

ax2 = Axis(  f[1,2],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            xlims = (-2.5,2.5),
            ylims = (-2.5,2.5))

for i in 1:curve_count
    lines!(ax2, solution_curves[:,1,i], solution_curves[:,2,i], color=:black)
end



################################
###### Hopf Bifurcaton #########
################################

function hopf!(du,u,μ,t)
    x,y = u
    du[1] = x * (1.0 - (x^2 + y^2)) - y
    du[2] = y * (1.0 - (x^2 + y^2)) + x
end

curve_count = 20
u0 = 4 * rand(mrng,2,curve_count) .- 2.0

solution_curves = zeros(201, 2, curve_count)

for i in 1:curve_count
    tspan = (0.0, 10.0)
    prob = ODEProblem(hopf!, u0[:,i], tspan, 1)

    sol = solve(prob, saveat=0.05)
    solution_curves[:, 1, i] = sol[1,:]
    solution_curves[:, 2, i] = sol[2,:]
end

ax3 = Axis(  f[1,3],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            xlims = (-2.0,2.0),
            ylims = (-2.0,2.0))

for i in 1:curve_count
    lines!(ax3, solution_curves[:,1,i], solution_curves[:,2,i], color=:black)
end



###############################
####### Lorenz System #########
###############################

function lorenz!(du,u,μ,t)
    x, y, z = u
    du[1] = 10.0 * (y - x)
    du[2] = x*(28.0 - z) - y
    du[3] = x * y  - 8.0/3.0 * z
end

curve_count = 3
u0 = 100 * rand(mrng, 3,curve_count) .- 50.0

solution_curves = zeros(2001, 3, curve_count)

for i in 1:curve_count
    tspan = (0.0, 10.0)
    prob = ODEProblem(lorenz!, u0[:,i], tspan, 1)

    sol = solve(prob, saveat=0.005)
    solution_curves[:, 1, i] = sol[1,:]
    solution_curves[:, 2, i] = sol[2,:]
    solution_curves[:, 3, i] = sol[3,:]
end

ax4 = Axis( f[1,4],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "z",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

for i in 1:curve_count
    lines!(ax4, solution_curves[:,1,i], solution_curves[:,3,i], color=:black)
end


##############################
######## Heat Equation #######
############################3#

heat = zeros(100,200)
heat[:,1] = randn(mrng, 100) .+ 1
heat[50:60,1] .+= 4
heat[90:100,1] .+= 4
α = 0.2
for i in 2:200
    heat[:,i] += (1.0 - 2.0 * α) * heat[:,i-1]
    heat[2:end,i] += α .* heat[1:end-1,i-1]
    heat[1:end-1,i] += α .* heat[2:end,i-1]
    heat[1,i] += α * heat[end,i-1]
    heat[end,i] += α * heat[1,i-1]
end

ax5 = Axis( f[1,5],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="τ", ylabel = "x",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

hm = heatmap!(ax5, (1:200)./100.0, (1:100)./100.0, heat')
Colorbar(f[1,6],hm, ticklabelsize=tickfontsize)


save("tablePlot.pdf", f)
