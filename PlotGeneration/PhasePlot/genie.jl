include("helper.jl")

##############################
##### Initialize Figure ######
##############################
@info("Initializing Figure...")

noto_sans = "./resources/NotoSans-Regular.ttf"

tickfontsize    = 28
labelfontsize   = 32

f = Figure(font=noto_sans, resolution=(8000,800))


####################################
###### Plot Differential Form ######
####################################
@info("Plotting Differential Form...")

vanderpol_arrow(u) = vanderpol(u,1,1)
ax1 = Axis(f[1,1], aspect=1,
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)
u_grid = -4:.333:4

us = zeros(length(u_grid)^2)
uv = zeros(length(u_grid)^2)
i=1
for y in u_grid
    for x in u_grid
        du = vanderpol_arrow([x,y])
        du /= sqrt(sum(abs2,du))
        us[i], uv[i] = du
        global i += 1
    end
end

us = reshape(us,length(u_grid),length(u_grid))/4
uv = reshape(uv,length(u_grid),length(u_grid))/4


arrows!(    ax1,
            u_grid,
            u_grid,
            us,
            uv)
    
xlims!(ax1, (-4,4))
ylims!(ax1, (-4,4))


###############################
###### Plot Trajectories ######
###############################

ax2 = Axis( f[1,2], aspect=1,
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

# Four different initial conditions
u0 = zeros(2,4)
u0[2,:] = [-.1,.1,-4,4]

for i in 1:4
    μ = 1.0
    tspan = (0.0,20.0)

    prob = ODEProblem(vanderpol!,u0[:,i],tspan,μ)

    sol = solve(prob,saveat=0.05)

    lines!(ax2,sol, color=:black)
end
xlims!(ax2, (-4,4))
ylims!(ax2, (-4,4))


##########################################
############ Phase Plot ##################
##########################################
@info("Generating Phase Plot...")

# Determine period of limit cycle

let
  u0 = [1,1]
  μ = 1.0
  tspan = (0.0,20.0)
  
  prob = ODEProblem(vanderpol!,u0,tspan,μ)
  
  sol = solve(prob,saveat=0.01)
  
  
  # Compute the distances from the final point and get one cycle
  distance = zeros(2001-100)
  for i in 1:2001-100
      distance[i] = sum(abs2,sol[end] - sol[i])
  end
  ~,ind = findmin(distance)
  period = 2001-ind
  global positions = sol.u[end-period+1:end]
  global positions = cat(positions...,dims=2)
end


u_grid = -4:.01:4

results = zeros(length(u_grid)^2)

Threads.@threads for i in 1:length(u_grid)
  for j in 1:length(u_grid)
    y = u_grid[j]
    x = u_grid[i]
    results[j + (i-1)*length(u_grid)] = phasemap([x,y],positions)
  end
end

results = reshape(results,length(u_grid),length(u_grid))


ax3 = Axis( f[2,2], 
            aspect=1,
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

heatmap!(ax3, u_grid, u_grid, results, colormap=:phase)
contour!(ax3, u_grid, u_grid, results, levels=31, color=:black)
xlims!(ax3,(-4,4))
ylims!(ax3,(-4,4))


#####################################
###### Generate Arrow Plot ##########
#####################################
@info("Plotting Informative Measurements...")

u_grid = -4:.3333:4
out_u = zeros(length(u_grid),length(u_grid))
out_v = zeros(length(u_grid),length(u_grid))

Threads.@threads for i in 1:length(u_grid)
    for j in 1:length(u_grid)
        u₀ = [u_grid[i], u_grid[j]]
        dx = ForwardDiff.jacobian(simsystem,u₀)
        ~, Σ, Vt = svd(dx)
        # Force counterclockwise
        test_vec = [u_grid[j], -u_grid[i]]
        if dot(test_vec,Vt[1,:]) < 0
            out_u[i,j] = Vt[1,1]
            out_v[i,j] = Vt[1,2]
        else
            out_u[i,j] = -Vt[1,1]
            out_v[i,j] = -Vt[1,2]
        end
    end
end


ax4 = Axis(f[2,1], aspect=1,
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize)

arrows!(    ax4,
            u_grid,
            u_grid,
            out_u./4,
            out_v./4)
    
xlims!(ax4, (-4,4))
ylims!(ax4, (-4,4))

##########################################
############### Save Plot ################
##########################################
@info("Saving Figure...")
save("out/demo.pdf",f);
