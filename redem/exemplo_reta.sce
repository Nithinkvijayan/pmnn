clc;
entrada = 'entrada109.txt';
desejada = 'desejada109.txt';
epocas_max = 10;
num_esp = 2;
eta = 0.3;
eta_gate = 0.37;
erro_max = 0.000001;
deb = 1;

[wt,at,et,ep] = redem_treinar(entrada,desejada,num_esp,eta,eta_gate,epocas_max,erro_max,deb);
j=1;
saida=0;

for p=-1:0.15:1
  saida(j) = redem_executar(p,wt,at);
  j = j + 1;
end

scf(1);
plot2d([-1:0.15:1]',saida,style=[5]);
scf(2);
plot2d([1:ep]',et,style=[5]);

