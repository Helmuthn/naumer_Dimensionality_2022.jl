using CSV
using LinearAlgebra: dot, I
using CairoMakie
using naumer_ICML_2022

const Σ = [1.0  0.1  0.1;
           0.1  1.0  0.1;
           0.1  0.1  1.0]

const v = [1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)]
const σ² = 1

f(x) = abs2(dot(v, Σ * x)) / (dot(x, Σ * x) + σ² * dot(x, x))


# Helper function to compute extra coordinate
function getZ(x,y)
    if abs2(x) + abs2(y) > 1 - 10*eps()
        return NaN
    end
    return sqrt(1 - abs2(x) - abs2(y))
end

function rewardPositive(x,y)
    return f([x,y,getZ(x,y)])
end

function rewardNegative(x,y)
    return f([x,y,-getZ(x,y)])
end


state = zeros(3,100)
state[3,1] = 1

for i in 2:100
    state[:,i] .= actiongradientstep_1DApprox(state[:,i-1], v, Σ, σ², 0.1)
end

rewards = zeros(100)
for i in 1:100
    rewards[i] = f(state[:,i])
end

# Sample Circle
x = -1.1:0.01:1.1
y = -1.1:0.01:1.1

noto_sans = "./resources/NotoSans-Regular.ttf"
noto_sans_bold = "./resources/NotoSans-Bold.ttf"

tickfontsize    = 26
labelfontsize   = 28
basewidth = 2

fig = Figure(font=noto_sans, resolution=(1000,500), figure_padding=40)


ax1 = Axis( fig[1,1],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            titlesize=labelfontsize,
            ratio=AxisAspect(1))


hm = heatmap!(ax1,x,y,rewardPositive)
lines!(ax1, state[1,:], state[2,:], color=:black, linewidth=3)


ax2 = Axis( fig[1,3],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="Iteration", ylabel = "Objective",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            titlesize=labelfontsize,
            ratio=AxisAspect(1))


lines!(ax2,1:100, rewards, color=:black, width=2) 

Colorbar(fig[1, 2], hm, ticklabelsize=tickfontsize)

colgap!(fig.layout,2,80)

Label(fig[1,1,TopLeft()], "A", font=noto_sans_bold, textsize = 36, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(fig[1,3,TopLeft()], "B", font=noto_sans_bold, textsize = 36, halign=:left, valign=:bottom, padding=(25,0,10,0))


save("out/OptDescent.pdf",fig)
