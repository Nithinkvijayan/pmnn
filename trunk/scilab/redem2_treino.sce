//
// UFRN - Universidade Federal do Rio Grande do Norte
// PPgEE - Programa de Pos-graduação em Engenharia Eletrica
//
// Rafael Marrocoso Magalhaes
//
// Codigo que implementa uma rede neural modular do tipo
// Modelo de Mistura Gaussiano Associativo com MLPs
// 
// Data criacao: 25 Out 2006
// Ultima alteracao: 2006 Out 25
//

// ////////////////////////////////////
// FUNCAO PARA O TREINAMENTO DA REDE M2
// ////////////////////////////////////

function [w,a,epoca,vet_erro] = redem2_treino(entrada,desejada,num_esp,arq_esp,func_esp,eta,bias,alfa,arq_gate,func_gate,eta_gate,bias_gate,alfa_gate,epocas_max,erro_max)

// ------------------
// LIMPANDO VARIAVEIS
// ------------------

  x=0; y=0; p=0; q=0;
  qt_pontos=0; qt_pontosd=0;
  w=0; a=0; d=0;
  erro_inst=0; tmom=0; soma_grad=0;
  soma_erro=0;

// ---------------------------
// LEITURA DO ARQUIVO DE DADOS
// ---------------------------
  
    // leitura dos dados de entrada
  x = fscanfMat(entrada);
  
    // leitura das respectivas respostas desejadas
  d = fscanfMat(desejada);
  
    // quant de pontos de treinamento e o tamanho de cada exemplo
  [qt_pontos, p] = size(x);
 
    // quant de pontos de treinamento deve ser igual a qpontos e tamanho de cada exemplo desejado
  [qt_pontosd, q] = size(d);

    // Verifica o tamanho dos arquivos
  if qt_pontos <> qt_pontosd
    disp('erro com os arquivos de treinamento: tamanho incompativel');
    abort
  end
  
// -------------  
// INICIALIZACAO
// -------------

    // sobre a arquitetura dos especialistas
  num_cam_esp = length(arq_esp);
  
    // sobre a arquitetura da passagem
  num_cam_gate = length(arq_gate);
  
    // quantidade maxima de neurônios por camada esp
  max_neu_esp = max(arq_esp);

    // quantidade maxima de neurônios por camada de passagem
  max_neu_gate = max(arq_gate);

    // pesos dos modulos, matriz de 4 dimensoes
    // (neuronios camada i + bias), (neuronios camada j),
    // (total de camadas), (cada especialista)
    // dwa delta atualizacao anterior para momento
  w = zeros((max_neu_esp+1), max_neu_esp, (num_cam_esp-1), num_esp);
  dwa = w;
  
    // pesos da rede de passagem, matriz de 3 dimensoes
    // (neu camada i + bias), (neuronios camada j),
    // (total de camadas)
  a = zeros((max_neu_gate+1), max_neu_gate, (num_cam_gate-1));
  daa = a;
  
    // funcional dos neuronios, matriz de 3 dimensoes
    // neuronios, camadas, especialistas
  yi = zeros(max_neu_esp, num_cam_esp, num_esp);

  
    // funcional dos neuronios, matriz de 2 dimensoes
    // neuronios, camadas
  ai = zeros(max_neu_gate, num_cam_gate);

    // campo local induzido de todos os neuronios
    // neuronios camadas especialistas
    // campo da rede de passagem
  campo = zeros(max_neu_esp, num_cam_esp, num_esp);
  campo_gate = zeros(max_neu_gate, num_cam_gate);
  
    // gradiente local de cada neuronio
    // neuronios, camadas, especialistas
    // gradiente da rede de passagem
  gradi = zeros(max_neu_esp, num_cam_esp, num_esp);
  gradi_gate = zeros(max_neu_gate, num_cam_gate);
  
    // armazena o valor da saída da rede
  y = zeros(num_esp,1);
  
    // probabilidae a priori
  g = zeros(num_esp,1);
  
    // probabilidade a posteriori
  h = zeros(num_esp,1);

    // valor do erro instantaneo, termo do momento
    // soma gradiente local, soma erro local
    // erro medio = valor superior para entrar no laco
  erro_inst = zeros(qt_pontos,1); tmom = 0;
  soma_gradi = 0; soma_erro=0;
  soma_campo = 0;
  erro_medio = erro_max + 10;
  vet_erro = zeros(epocas_max,1);

    // valor do erro local, num neuronios saida, num especialistas
  erro_local = zeros(arq_esp(num_cam_esp), num_esp);
  
  

// -----------------------
// INICIALIZACAO DOS PESOS
// -----------------------
  
    // inicializacao dos pesos sinapticos w com valores de 0 a 0.5
  w = rand(w);
  w = w*0.5;
  
    // inicializacao dos pesos sinapticos a com valores de 0 a 0.5
  a = rand(a);
  a = a*0.5;
  
    // Indice para acesso ao vetor de entrada
  indice = [1:qt_pontos];
  //indice = rand_indices(indice);
  

// -------------------------------
// INICIO ALGORITMO DE TREINAMENTO
// -------------------------------

    // epocas de treinamento
  epoca = 0;

    // INICIO LACO DE TREINAMENTO
    // enquanto nao atingir o erro ou limite de epocas
  while ( (erro_max < erro_medio) & (epocas_max > epoca) )
  
      // condicao de parada por explosao do valor do erro
    if ( (erro_medio > 10^10 ) & ( epoca > 100 ) )
      break;
    end
  
      // zera o valor do erro medio quadrado
    erro_medio = 0;
    
      // INICIO DE UMA EPOCA DE TREINAMENTO
    for exemplo=1:qt_pontos
    
        // ---------------------
        // 1 - PASSO PARA FRENTE
        // ---------------------
        
        // para cada especialista
        // ----------------------
      for esp=1:num_esp
      
          // para cada camada dos especialistas
        for camada=1:num_cam_esp
        
            // caso especial de ser a primeira camada
          if camada == 1
            
              // para cada neuronio da camada de entrada
            for j=1:arq_esp(1)
              campo(j,camada,esp) = x(indice(exemplo),j);
              yi(j,camada,esp) = x(indice(exemplo),j);              
            end
            
          end // caso primeira camada
          
            // caso camada diferente da entrada
          if camada > 1
            
              // para cada neuronio da camada
            for j=1:arq_esp(camada)
              
              soma_campo = 0; // somatorio do campo local induzido
              
                // adicao do bias e seu peso
              soma_campo = soma_campo + ( w(1,j,(camada-1),esp) * bias );
              
                // atualizacao do campo devido neuronios anteriores
              for i=1:arq_esp(camada-1)
                soma_campo = soma_campo + ( w((i+1),j,(camada-1),esp) * yi(i,(camada-1),esp) );
              end
              
                // campo local induzido
              campo(j,camada,esp) = soma_campo;
              
                // saida do neuronio
              yi(j,camada,esp) = rna_funcao_ativacao( soma_campo, func_esp(camada) );
              
            end
          
          end // fim camada
          
        end // fim para camada espec
        
      end // fim para especialistas
      
        // para a rede de passagem
        // -----------------------        
      for camada=1:num_cam_gate
      
            // caso especial de ser a primeira camada
          if camada == 1
            
              // para cada neuronio da camada de entrada
            for j=1:arq_gate(1)
              campo_gate(j,camada) = x(indice(exemplo),j);
              ai(j,camada) = x(indice(exemplo),j);
            end
          
          end // caso primeira camada
          
            // caso camada diferente da entrada
          if camada > 1
            
              // para cada neuronio da camada
            for j=1:arq_gate(camada)
              
              soma_campo = 0; // somatorio do campo local induzido
              
                // adicao do bias e seu peso
              soma_campo = soma_campo + ( a(1,j,(camada-1)) * bias );
              
                // atualizacao do campo devido neuronios anteriores
              for i=1:arq_gate(camada-1)
                soma_campo = soma_campo + ( a((i+1),j,(camada-1)) * ai(i,(camada-1)) );
              end
              
                // campo local induzido
              campo_gate(j,camada) = soma_campo;
              
                // saida do neuronio
              ai(j,camada) = rna_funcao_ativacao( soma_campo, func_gate(camada) );
              
            end
          
          end // fim camada
          
      end // fim para rede de passagem

      
        // calculo das probabilidades a priori
        // -----------------------------------
      gsoma = 0; // somatorio das exp de gi
      gexp = zeros( num_esp, 1); // exp de gi
      
      for i=1:num_esp
        gexp(i) = exp(ai(i, num_cam_gate));
        gsoma = gsoma + gexp(i);
      end
      
      for i=1:num_esp
        g(i) = gexp(i)/gsoma;
      end
        
        // obtencao da saida da rede
        // -------------------------
      y = zeros(q,1);
      for i=1:q
        for esp=1:num_esp
          y(i) = y(i) + ( yi(i,num_cam_esp,esp) * g(esp) );
        end
      end
      
      
        // Calculo dos erros
        // -----------------
      for esp=1:num_esp
        for i=1:q
          erro_local(i,esp) = d(indice(exemplo),i) - yi(i,num_cam_esp,esp);
          soma_erro = soma_erro + erro_local(i,esp);
        end
      end
      
      erro_inst(exemplo) = soma_erro/2;
      
            
        // Calculo da prob a posteriori
        // ----------------------------
      hsoma = 0; // somatorio das exp de hi
      hexp = zeros( num_esp, 1); // exp de hi
      
      for esp=1:num_esp
        temp = 0;
        for i=1:q
          temp = temp + erro_local(i,esp);
        end
        hexp(esp) = g(esp) * exp( -0.5 * (temp^2) );
        hsoma = hsoma + hexp(esp);
      end
      
      for esp=1:num_esp
        h(esp) = hexp(esp)/hsoma;
      end
      
        
        // -------------------
        // 2 - PASSO PARA TRAS
        // -------------------
      
        // Ajuste dos pesos dos especialistas
        // ----------------------------------

        // para cada especialista
        // ----------------------
      for esp=1:num_esp
      
          // vai da ultima cama ate a primeira
        for camada=num_cam_esp:-1:2
        
            // caso especial camada de saida
          if camada == num_cam_esp
          
              // para cada neuronio da saida
            for j=1:q
            
                // calculo do gradiente local
              gradi(j,camada,esp) = erro_local(j,esp) * rna_funcao_deriv_ativacao( campo(j,camada,esp), func_esp(camada) );
                // calculo do deltaw * UTILIZA-SE O H *
              deltaw = eta * gradi(j,camada,esp) * bias * h(esp);
                // termo do momento
              aux = w(1,j,(camada-1),esp);
              mom = alfa * dwa(1,j,(camada-1),esp);
              
                // atualizacao do peso do bias
              w(1,j,(camada-1),esp) = aux + mom + deltaw;
              dwa(1,j,(camada-1),esp) = deltaw;
              
                // percorre todos os pesos com neuronios anteriores
              for i=1:arq_esp(camada-1)
                deltaw = eta * gradi(j,camada,esp) * yi(i,(camada-1),esp) * h(esp);
                aux = w((i+1),j,(camada-1),esp);
                mom = alfa * dwa((i+1),j,(camada-1),esp);
                w((i+1),j,(camada-1),esp) = aux + mom + deltaw;
                dwa((i+1),j,(camada-1),esp) = deltaw;
              end
              
            end // fim neuronio
            
          end // fim camada de saida
          
            // para cada camada oculta
          if camada < num_cam_esp
          
              // para cada neuronio
            for j=1:arq_esp(camada)
            
              soma_gradi = 0;
                // para cada neuronio da camada da frente
              for k=1:arq_esp(camada)
                soma_gradi = soma_gradi + ( gradi(k,(camada+1),esp) * w(k,(j+1),camada,esp) );
              end
                // calculo do gradiente local
              gradi(j,camada,esp) = soma_gradi * rna_funcao_deriv_ativacao( campo(j,camada,esp), func_esp(camada) );  
                // calculo do deltaw * UTILIZA-SE O H *
              deltaw = eta * gradi(j,camada,esp) * bias;
                // termo do momento
              aux = w(1,j,(camada-1),esp);
              mom = alfa_gate * dwa(1,j,(camada-1));
              
                // atualizacao do peso do bias
              w(1,j,(camada-1),esp) = aux + mom + deltaw;
              dwa(1,j,(camada-1),esp) = deltaw;
              
                // percorre todos os pesos com neuronios anteriores
              for i=1:arq_esp(camada-1)
                deltaw = eta * gradi(j,camada,esp) * yi(i,(camada-1),esp);
                aux = w((i+1),j,(camada-1),esp);
                mom = alfa * dwa((i+1),j,(camada-1),esp);
                w((i+1),j,(camada-1),esp) = aux + mom + deltaw;
                dwa((i+1),j,(camada-1),esp) = deltaw;
              end
              
            end // fim neuronio
          
          end // fim camada oculta
        
        end // fim camada especialista
      
      end // fim para cada especialista
      
        // Atualizacao da rede passagem
        // ----------------------------
        
        // da ultima camada ate a primeira
      for camada=num_cam_gate:-1:2
      
          // caso especial camada de saida
        if camada == num_cam_gate
        
            // para cada neuronio da saida
          for j=1:q
          
              // calculo do gradiente local
            gradi_gate(j,camada) = rna_funcao_deriv_ativacao( campo_gate(j,camada), func_gate(camada) );
              // calculo do deltaw * UTILIZA-SE O H *
            deltaw = eta_gate * gradi_gate(j,camada) * ( h(esp) - g(esp) ) * bias_gate;
              // termo do momento
            aux = a(1,j,(camada-1));
            mom = alfa * daa(1,j,(camada-1));
            
              // atualizacao do peso do bias
            a(1,j,(camada-1)) = aux + mom + deltaw;
            daa(1,j,(camada-1)) = deltaw;
            
              // percorre todos os pesos com neuronios anteriores
            for i=1:arq_gate(camada-1)
              deltaw = eta_gate * gradi_gate(j,camada) * ( h(esp) -g(esp) ) * ai(i,(camada-1));
              aux = a((i+1),j,(camada-1));
              mom = alfa_gate * daa((i+1),j,(camada-1));
              a((i+1),j,(camada-1)) = aux + mom + deltaw;
              daa((i+1),j,(camada-1)) = deltaw;
            end
            
          end // fim neuronio
          
        end // fim camada de saida
        
          // para cada camada oculta
        if camada < num_cam_esp
        
            // para cada neuronio
          for j=1:arq_gate(camada)
          
            soma_gradi = 0;
              // para cada neuronio da camada da frente
            for k=1:arq_gate(camada)
              soma_gradi = soma_gradi + ( gradi_gate(k,(camada+1)) * a(k,(j+1),camada) );
            end
              // calculo do gradiente local
            gradi_gate(j,camada) = soma_gradi * rna_funcao_deriv_ativacao( campo_gate(j,camada), func_gate(camada) );  
              // calculo do deltaw * UTILIZA-SE O H *
            deltaw = eta * gradi_gate(j,camada) * bias_gate;
              // termo do momento
            aux = a(1,j,(camada-1));
            mom = alfa_gate * daa(1,j,(camada-1));
            
              // atualizacao do peso do bias
            a(1,j,(camada-1)) = aux + mom + deltaw;
            daa(1,j,(camada-1)) = deltaw;
            
              // percorre todos os pesos com neuronios anteriores
            for i=1:arq_gate(camada-1)
              deltaw = eta * gradi_gate(j,camada) * ai(i,(camada-1));
              aux = a((i+1),j,(camada-1));
              mom = alfa_gate * daa((i+1),j,(camada-1));
              a((i+1),j,(camada-1)) = aux + mom + deltaw;
              daa((i+1),j,(camada-1)) = deltaw;
            end
            
          end // fim neuronio
        
        end // fim camada oculta
      
      end // fim camada gate
      
        // -------------------
        // 3 - CALCULO DO ERRO
        // -------------------
        
        // atualiza o valor da media do erro quadratico medio do treinamento
      temp = 0;
      
      for l=1:qt_pontos
        temp = temp + erro_inst(l);
      end
      
      erro_medio = temp / qt_pontos;
      
        // incrementa a epoca
      epoca = epoca + 1;
      
        // adiciona o valor do erro_medio no vetor de erro
      vet_erro(epoca) = erro_medio;
      
      // ajusta os indices dos dados de treinamento para proxima epoca
      indice = rand_indice(indice);
    
      
    end // FIM DE UMA EPOCA DE TREINAMENTO
  
  end // FIM LACO DE TREINAMENTO

  
endfunction


// ////////////////////////////////////
// Funcao de ativacao para os neuronios
// ////////////////////////////////////
function [y] = rna_funcao_ativacao( x, tipo )

    // Funcao linear
  if tipo == 0
  	y = x;
  end

   // Funcao Sigmoide
  if tipo == 1
  	y = 1/( 1 + exp(-x) );
  end

   // Funcao Tangente Hiperbolia
  if tipo == 2
  	a = 1.7159;
  	b = 2/3;
  	y = a*tanh(b*x);
  end

   // Funcao de limiar para 0.5
  if tipo == 3
  	if x < 0.5
  		y = 0;
  	else
  		y = 1;
  	end
  end

endfunction


// /////////////////////////////////////////////
// Funcao derivada de ativacao para os neuronios
// /////////////////////////////////////////////

function [y] = rna_funcao_deriv_ativacao( x, tipo )

   // Funcao linear
  if tipo == 0
  	y = 1;
  end

    // Funcao Sigmoide
  if tipo == 1
  	y = (1/( 1 + exp(-x) )) - (1/( 1 + exp(-x) ))^2;
  end

    // Funcao Tangente Hiperbolia
  if tipo == 2
  	a = 1.7159;
  	b = 2/3;
  	y = a*b*( 1 - (tanh(b*x))^2 );
  end

    // Funcao de limiar para 0.5
  if tipo == 3
  	if x < 0.5
  		y = 0;
  	else
  		y = 1;
  	end
  end

endfunction


// //////////////////////////////////////////////////////////////
// Funcao para randomizar o indice de acesso aos dados de entrada
// //////////////////////////////////////////////////////////////
function [vet_indice] = rand_indice(vet_indice)

    // quantidade de pontos do indice
  qt_pontos = length(vet_indice)

    // cria vetor temporaria para a nova ordem do indice
  novo_ind = zeros(qt_pontos,1);
    
    // gera a ordem de troca dos dados
  for i=1:qt_pontos
      novo_ind(i) = floor(rand()*qt_pontos + 1);
  end

    // cria a nova ordem no indice vet_indice()
  for i=1:qt_pontos
      temp = vet_indice(novo_ind(i));
      vet_indice(novo_ind(i)) = vet_indice(i);
      vet_indice(i) = temp;
    end

endfunction



