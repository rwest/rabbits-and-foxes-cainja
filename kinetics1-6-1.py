import matplotlib.pyplot as plt

x = 500
y = 200
k1 = 0.02
k2 = 0.00004
k3 = 0.0004
k4 = 0.04

xlist = [x]
ylist = [y]
dxlist = []
dylist = []
tlist = [0]

for t in range(500*24*60):
    dx = (k1*x) - (k2*x*y)
    dy = (k3*x*y) - (k4*y)
    x += dx*1/24/60
    y += dy*1/24/60
    if t%(24*60) == 0: #clean up the plots a little bit
        xlist.append(x)
        ylist.append(y)
        tlist.append((t+1)/24/60)

fig, ax = plt.subplots(1,1)
rabbits, = ax.plot(tlist, xlist, '.', markersize = 5, label = 'Rabbits')
foxes, = ax.plot(tlist, ylist, '^', markersize = 5, label = 'Foxes' )
ax.legend(loc=7) #move the legend to above the graph
ax.set_title('Rabbits and Foxes v. Time')
ax.set_xlabel('Time (day)')
ax.set_ylabel('Number')

fig.tight_layout()
plt.show()
