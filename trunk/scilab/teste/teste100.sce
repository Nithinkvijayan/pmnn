//treinamento da rede
clc;

entrada = 'entrada100.txt';
desejada = 'desejada100.txt';

num_esp = 2;
arq_esp = [1,1];
func_esp = [0,0];
bias = 0;
eta = 0.026;
alfa = 0;


arq_gate = [1,2];
func_gate = [0,0];
bias_gate = 0;
eta_gate = 0.039;
alfa_gate = 0;

epocas_max = 300;

erro_max = 0.00001;
//erro_max = 1000;

[w,a,qt_epocas,vet_erro] = redem2_treino(entrada,desejada,num_esp,arq_esp,func_esp,eta,bias,alfa,arq_gate,func_gate,eta_gate,bias_gate,alfa_gate,epocas_max,erro_max);


//[y] = redem2_executar(entrada,num_esp,arq_esp,w,func_esp,bias,arq_gate,a,func_gate,bias_gate)
j=1;
saida=0;

for p=-1:0.15:1
  saida(j) = redem2_executar(p,num_esp,arq_esp,w,func_esp,bias,arq_gate,a,func_gate,bias_gate);
  j = j + 1;
end

scf(1);
plot2d([-1:0.15:1]',saida,style=[5]);
scf(2);
plot2d([1:qt_epocas]',vet_erro,style=[5]);
