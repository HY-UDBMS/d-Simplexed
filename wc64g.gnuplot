set terminal png size 1024,768
set output "wc64g.png"

set title "Wordcount 64GB Test Cases Runtime"
set xlabel "Virtual Memory (GB)" center rotate by -55
set ylabel "Virtual Cores" rotate by 90
set zlabel "Duration (s)" center rotate by 90

set dgrid3d 50,10,10
set hidden3d
#set pm3d 

set view 60, 75, 1, 1

splot "data/wc64g_v2.dat" u 1:2:3 with lines
