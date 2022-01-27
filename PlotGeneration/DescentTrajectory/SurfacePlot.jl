using CSV
using LinearAlgebra: dot, I
using CairoMakie

const Σ = [1.0  0.1  0.1;
           0.1  1.0  0.1;
           0.1  0.1  1.0]

const v = [1.0/sqrt(3), 1.0/sqrt(3), 1.0/sqrt(3)]
const σ² = 1

const Σ2 = -0.95 * v * v' + I(3)

f(x) = abs2(dot(v, Σ * x)) / (dot(x, Σ * x) + σ² * dot(x, x))

g(x) = abs2(dot(v, Σ2 * x)) / (dot(x, Σ2 * x) + σ² * dot(x, x))

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


function rewardPositive2(x,y)
    return g([x,y,getZ(x,y)])
end

function rewardNegative2(x,y)
    return g([x,y,-getZ(x,y)])
end

# Sample Circle
x = -1.1:0.01:1.1
y = -1.1:0.01:1.1

noto_sans = "./resources/NotoSans-Regular.ttf"
noto_sans_bold = "./resources/NotoSans-Bold.ttf"

tickfontsize    = 26
labelfontsize   = 28
basewidth = 2

fig = Figure(font=noto_sans, resolution=(1500,525), figure_padding=40)


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


ax2 = Axis( fig[1,2],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            titlesize=labelfontsize,
            ratio=AxisAspect(1))


heatmap!(ax2,x,y,rewardNegative)


ax3 = Axis( fig[1,3],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            titlesize=labelfontsize,
            ratio=AxisAspect(1))


hm2 = heatmap!(ax3,x,y,rewardPositive2)


ax4 = Axis( fig[1,4],
            xticklabelsize=tickfontsize, 
            yticklabelsize=tickfontsize, 
            yticklabelpad=2,
            xlabel="x", ylabel = "y",
            xlabelsize=labelfontsize,
            ylabelsize=labelfontsize,
            titlesize=labelfontsize,
            ratio=AxisAspect(1))


heatmap!(ax4,x,y,rewardNegative2)


Colorbar(fig[2, 1:2], hm, vertical = false, flipaxis = false, ticklabelsize=tickfontsize)
Colorbar(fig[2, 3:4], hm2, vertical = false, flipaxis = false, ticklabelsize=tickfontsize)

colgap!(fig.layout,15)
colgap!(fig.layout,2,100)

Label(fig[1,1,TopLeft()], "A", font=noto_sans_bold, textsize = 36, halign=:left, valign=:bottom, padding=(25,0,10,0))
Label(fig[1,3,TopLeft()], "B", font=noto_sans_bold, textsize = 36, halign=:left, valign=:bottom, padding=(25,0,10,0))


save("out/OptSurface.pdf",fig)
