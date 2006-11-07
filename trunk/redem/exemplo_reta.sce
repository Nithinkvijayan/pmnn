clc;
num_epocas = 400;
num_esp = 2;
eta = 0.026;
eta2 = 0.039;
[wt,at,et] = redem_treino('entrada108.txt','desejada108.txt',num_esp,num_epocas,eta,eta2);

j=1;
saida=0;

for p=-1:0.15:1
  saida(j) = redem_run(p,wt,at);
  j = j + 1;
end

scf(1);
plot2d([-1:0.15:1]',saida,style=[5]);
scf(2);
plot2d([1:num_epocas]',et,style=[5]);

