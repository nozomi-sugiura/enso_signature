set term png
set grid
unset xrange 
set xrange [1980:2022]
unset yrange 
k = 0;
while (k < 12) {
stats "< cat ts_ord3_??????.txt" u 1:($2-$3)**2 ev 12::k nooutput; r1=STATS_mean_y**0.5;
print 'RMSE FOR MON ', k+1, r1;
k=k+1;
}

stats "< cat ts_ord3_??????.txt" u 1:($2-$3)**2 nooutput; r1=STATS_mean_y**0.5
print 'RMSE FOR ALL - ', r1


set terminal epscairo color enhanced font "Helvetica,18" #background rgb "gray"
set key right bottom
set size ratio 0.5
set output "pred.eps"
plot \
"< cat ts_ord3_??????.txt" u 1:($2+$4)  title "obs" w l lw 3 lc rgb "black", \
"< cat ts_ord3_??????.txt" u 1:($3+$4)  title "pred" w l lw 2 lc rgb "red", \



