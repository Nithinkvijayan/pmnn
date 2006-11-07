clc;
[wt,at] = redem_treino('entrada100.txt','desejada100.txt',2,100,0.03,0.03);

j=1;

for p=-1:0.05:1
  saida(j) = redem_run(p,wt,at);
  j = j + 1;
end

plot2d([-1:0.05:1],saida);


