using CairoMakie
using DifferentialEquations

noto_sans = "./resources/NotoSans-Regular.ttf"

tickfontsize    = 28
labelfontsize   = 32


function vanderpol!(du,u,μ,t)
    x,y = u
    du[1] = μ * (x - x^3/3 - y)
    du[2] = x/μ
end

function vanderpol(u,μ,t)
    du = zeros(2)
    vanderpol!(du,u,μ,t)
    return du
end

#%%
f = Figure(font=noto_sans, resolution=(800,400))

ax = Axis(  f[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Time", ylabel = "State",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

curve_count = 100
u0 = 2*randn(2,curve_count)

solution_curves = zeros(401,2,curve_count)
timings = 0:0.05:20

for i in 1:curve_count
    μ = 2.0
    tspan = (0.0,20.0)

    prob = ODEProblem(vanderpol!,u0[:,i],tspan,μ)

    sol = solve(prob,saveat=0.05)
    solution_curves[:,1,i] = sol[1,:]
    solution_curves[:,2,i] = sol[2,:]

end

# Find the last peak
peaks = zeros(Integer,curve_count)
for i in 1:curve_count
    ~, peaks[i] = findmax(solution_curves[250:400,1,i])
end

# Now actually plot finally
for i in 1:curve_count
    lines!(ax,timings[1:100],solution_curves[peaks[i]:peaks[i]+100-1,1,i], color=:black)
    lines!(ax,timings[1:100],solution_curves[peaks[i]:peaks[i]+100-1,2,i], color=:blue)
end
xlims!(ax, (0,5.0))
ylims!(ax, (-4,4))
f

save("converge.pdf",f)
